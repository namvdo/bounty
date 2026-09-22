import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import * as THREE from 'three';
import { Line2 } from 'three/examples/jsm/lines/Line2.js';
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import { Shell } from './components/layout/Shell';
import { Sidebar } from './components/layout/Sidebar';
import { Viewport } from './components/layout/Viewport';
import { normalizeParams } from './utils/paramUtils';
import {
    DEFAULT_VIEW_RANGE,
    DEFAULT_VIEWPORT_RANGE,
    RANGE_LIMIT,
    ZOOM_IN_FACTOR,
    ZOOM_OUT_FACTOR,
    constrainViewRange,
    normalizeViewRange,
    orthographicFrustumSize,
    zoomViewRange
} from './utils/viewRange';
import {
    DEFAULT_PERIODIC_SEARCH_SETTINGS,
    forceFullGridSearchSettings,
    normalizePeriodicSearchSettings
} from './utils/periodicSearchSettings';
import { applyStartPointUpdate } from './utils/startPointState';
import {
    appendTrajectoryHistoryPoint,
    shouldRecordTrajectoryHistoryPoint
} from './utils/trajectoryState';
import {
    buildVerifiedBoundaryCycles,
    collectExtendedManifoldBranches,
} from './utils/geometricOffsetSeed';
import {
    BOUNDARY_LAYER_COLORS,
    reconstructDeterministicImageBoundary,
    summarizeOrderedBoundaryBranches,
} from './utils/boundaryLayers';
import {
    buildGeometricOffsetBatchRequest,
    geometricOffsetSampleSpacing
} from './utils/geometricOffsetCompute';
import {
    createGeometricOffsetContour,
    geometricOffsetSourceContours,
    reconcileGeometricOffsetContours
} from './utils/geometricOffsetBatch';
import {
    fitInverseOffsetCurveRange,
    inverseOffsetCurveColor,
    visibleInverseOffsetCurves
} from './utils/inverseOffsetDisplay';
import {
    computeContourVertexColors,
    computeInverseCurveVertexColors,
} from './utils/inverseOffsetColors';
import {
    enrichSolutionPointsWithOrbitNormals,
    fixedPointSolutionsFromOrbits,
    orbitExtendedStates
} from './utils/extendedOrbitState';
import {
    applyParameterAnimationValue,
    beginPeriodicRefresh,
    capturePeriodicSearchSettings,
    isParameterAnimationStepSettled,
    nextParameterAnimationStep
} from './utils/parameterAnimationSync';
import { useComputeWorker } from './hooks/useComputeWorker';
import {
    INITIAL_CUSTOM_EQUATIONS,
    INITIAL_CUSTOM_PARAMS,
    INITIAL_PARAMS,
    SYSTEM_CATALOG,
    isCustomSystem as isCustomSystemId,
    supportsPeriodicSearch,
    systemTypeFor
} from './config/systems';
import {
    CONTINUOUS_BOUNDARY_FLOW_SETTINGS,
    DEFAULT_GEOMETRIC_OFFSET_SETTINGS,
    DEFAULT_INVARIANT_SET_SETTINGS,
    DEFAULT_MANIFOLD_SETTINGS,
    DEFAULT_ULAM_SETTINGS,
    INVERSE_OFFSET_POSITION_TOLERANCE_RULE,
    ULAM_OPERATOR_SETTINGS,
    continuousUlamIntegrationTime,
    inverseOffsetPositionTolerance
} from './config/numericalSettings';
import { randomHenonInvariantSeed } from './utils/invariantSetSeed';
import {
    buildExperimentBundle,
    experimentConfigurationToUiState,
    parseExperimentBundle
} from './utils/experimentBundle';
import type { ParameterValidation } from './utils/paramUtils';
import type { SupportGrid } from './protocol/computeContracts';
import type {
    AnimationState,
    BdeState,
    BistParameters,
    BistWasmModule,
    CustomEquation,
    CustomEquations,
    CustomParameter,
    CustomParameters,
    ExperimentStatus,
    ExtendedPointTuple,
    ExtendedState,
    GeometricOffsetState,
    InvariantSetState,
    Manifold,
    ManifoldBranch,
    ManifoldState,
    OrbitFilters,
    PeriodicOrbit,
    PeriodicSearchSettings,
    PeriodicState,
    ProjectedState,
    RecordingState,
    SolutionPoint,
    SweepState,
    SystemId,
    SystemType,
    TooltipState,
    UlamBox,
    UlamState,
    UlamTransition,
    UnknownRecord,
    ViewRange,
} from './types/domain';

const GRID_STYLE = {
    gridDivisions: 16,
    axisColor: 0x888888,
    gridColor: 0x333333
};

const EMPTY_ARRAY: never[] = [];
const DEFAULT_VALIDATION: ParameterValidation = { normalized: [], errors: [], valid: true };
const PARAM_ABS_LIMIT = 10;

const pointInDomain = (point: { x: number; y: number }, domain: ViewRange): boolean => (
    Number.isFinite(point.x)
    && Number.isFinite(point.y)
    && point.x >= domain.xMin
    && point.x <= domain.xMax
    && point.y >= domain.yMin
    && point.y <= domain.yMax
);

type BdeSimulator =
    | InstanceType<BistWasmModule['BdeSimulatorWasm']>
    | InstanceType<BistWasmModule['BdeSimulatorUserDefinedWasm']>;

declare global {
    interface Window {
        update_start_point?: (x: number, y: number, nx: number, ny: number) => void;
    }
}

const projectedCoordinates = (
    point: ProjectedState | ExtendedState | ExtendedPointTuple,
): { x: number; y: number } | null => {
    const x = Array.isArray(point) ? point[0] : point.x;
    const y = Array.isArray(point) ? point[1] : point.y;
    return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null;
};

const getSupportIndex = (x: number, y: number, support: SupportGrid | null): number => {
    if (!support) return -1;
    const { xMin, xMax, yMin, yMax, subdivisions } = support;
    if (x < xMin || x > xMax || y < yMin || y > yMax) return -1;

    const dx = (xMax - xMin) / subdivisions;
    const dy = (yMax - yMin) / subdivisions;

    if (!Number.isFinite(dx) || !Number.isFinite(dy) || dx <= 0 || dy <= 0) {
        return -1;
    }

    let ix = Math.floor((x - xMin) / dx);
    let iy = Math.floor((y - yMin) / dy);
    if (ix >= subdivisions) ix -= 1;
    if (iy >= subdivisions) iy -= 1;
    if (ix < 0 || iy < 0) return -1;
    return iy * subdivisions + ix;
};

const getGridBoxIndex = (
    x: number,
    y: number,
    range: ViewRange,
    subdivisions: number,
): number => {
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(subdivisions) || subdivisions <= 0) {
        return -1;
    }
    if (x < range.xMin || x > range.xMax || y < range.yMin || y > range.yMax) return -1;

    const dx = (range.xMax - range.xMin) / subdivisions;
    const dy = (range.yMax - range.yMin) / subdivisions;
    if (!Number.isFinite(dx) || !Number.isFinite(dy) || dx <= 0 || dy <= 0) return -1;

    let ix = Math.floor((x - range.xMin) / dx);
    let iy = Math.floor((y - range.yMin) / dy);
    if (ix >= subdivisions) ix -= 1;
    if (iy >= subdivisions) iy -= 1;
    if (ix < 0 || iy < 0) return -1;
    return iy * subdivisions + ix;
};

const isSupportedPoint = (x: number, y: number, support: SupportGrid | null): boolean => {
    if (!support) return true;
    const idx = getSupportIndex(x, y, support);
    if (idx < 0) return false;
    return (support.invariantMeasure?.[idx] ?? 0) > support.threshold;
};

const clipTrajectoryBySupport = (
    traj: ManifoldBranch | undefined,
    support: SupportGrid | null,
): ManifoldBranch | undefined => {
    if (!traj?.points || !support) return traj;
    return {
        ...traj,
        points: traj.points.filter(([x, y]) => isSupportedPoint(x, y, support))
    };
};

const clipManifoldsBySupport = (manifolds: Manifold[], support: SupportGrid | null): Manifold[] => {
    if (!support) return manifolds || [];
    return (manifolds || [])
        .map(m => ({
            ...m,
            plus: clipTrajectoryBySupport(m.plus, support),
            minus: clipTrajectoryBySupport(m.minus, support)
        }))
        .filter(m => ((m.plus?.points?.length || 0) + (m.minus?.points?.length || 0)) > 0);
};

const clampToRange = (
    value: number,
    minValue: number,
    maxValue: number,
    fallbackValue: number,
): number => {
    if (!Number.isFinite(value)) {
        return fallbackValue;
    }
    if (value < minValue) return minValue;
    if (value > maxValue) return maxValue;
    return value;
};


const calculateNiceStep = (span: number): number => {
    const rawStep = Math.abs(span) / 4;
    const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const residual = rawStep / magnitude;

    let step: number;
    if (residual < 1.5) step = 1 * magnitude;
    else if (residual < 3.5) step = 2 * magnitude;
    else if (residual < 7.5) step = 5 * magnitude;
    else step = 10 * magnitude;

    return Number(step.toFixed(6));
};

const formatTickNumber = (num: number): string => {
    const rounded = Number(num.toFixed(6));
    if (Math.abs(rounded) < 1e-6) return '0';
    const str = rounded.toString();
    if (!str.includes('.')) return `${str}.0`;
    return str;
};

const createCoordinateSystem = (
    scene: THREE.Scene,
    range: ViewRange
): THREE.Group => {
    const { axisColor, gridColor } = GRID_STYLE;
    const limit = RANGE_LIMIT;

    const xMin = -limit;
    const xMax = limit;
    const yMin = -limit;
    const yMax = limit;

    const xSpan = Math.abs(range.xMax - range.xMin);
    const ySpan = Math.abs(range.yMax - range.yMin);

    const viewSpan = Math.min(xSpan, ySpan);
    const xTickStep = calculateNiceStep(xSpan);
    const yTickStep = calculateNiceStep(ySpan);

    const gridGroup = new THREE.Group();
    gridGroup.name = 'coordinate-system';

    const xGridStep = xTickStep / 2;
    const numX = Math.ceil(limit / xGridStep);
    for (let i = -numX; i <= numX; i++) {
        const x = Number((i * xGridStep).toFixed(6));
        if (Math.abs(x) > limit) continue;
        const isAxis = Math.abs(x) < 0.001;
        const isMajor = Math.abs(x % xTickStep) < 0.001 || Math.abs(Math.abs(x % xTickStep) - xTickStep) < 0.001;
        const points = [
            new THREE.Vector3(x, yMin, -0.01),
            new THREE.Vector3(x, yMax, -0.01)
        ];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
            color: isAxis ? axisColor : gridColor,
            transparent: true,
            opacity: isAxis ? 1.0 : isMajor ? 0.35 : 0.15
        });
        const line = new THREE.Line(geometry, material);
        line.userData.isGrid = true;
        gridGroup.add(line);
    }

    const yGridStep = yTickStep / 2;
    const numY = Math.ceil(limit / yGridStep);
    for (let i = -numY; i <= numY; i++) {
        const y = Number((i * yGridStep).toFixed(6));
        if (Math.abs(y) > limit) continue;
        const isAxis = Math.abs(y) < 0.001;
        const isMajor = Math.abs(y % yTickStep) < 0.001 || Math.abs(Math.abs(y % yTickStep) - yTickStep) < 0.001;
        const points = [
            new THREE.Vector3(xMin, y, -0.01),
            new THREE.Vector3(xMax, y, -0.01)
        ];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
            color: isAxis ? axisColor : gridColor,
            transparent: true,
            opacity: isAxis ? 1.0 : isMajor ? 0.35 : 0.15
        });
        const line = new THREE.Line(geometry, material);
        line.userData.isGrid = true;
        gridGroup.add(line);
    }

    const createTextSprite = (
        text: string,
        position: THREE.Vector3,
        fontSize = 0.15,
    ): THREE.Sprite => {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 128;
        canvas.height = 64;
        if (!context) throw new Error('Could not create the coordinate-label canvas context.');
        context.fillStyle = 'transparent';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.font = '700 28px Inter, Arial, sans-serif';
        context.fillStyle = '#aab2bd';
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText(text, canvas.width / 2, canvas.height / 2);

        const texture = new THREE.CanvasTexture(canvas);
        texture.minFilter = THREE.LinearFilter;
        const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
        const sprite = new THREE.Sprite(material);
        sprite.position.copy(position);
        sprite.scale.set(fontSize * 2.2, fontSize, 1);
        sprite.userData.isGrid = true;
        return sprite;
    };

    const labelFontSize = Math.min(0.12, Math.max(0.004, viewSpan * 0.028));
    const xLabelOffsetY = -Math.min(0.14, Math.max(0.005, ySpan * 0.035));
    const yLabelOffsetX = -Math.min(0.14, Math.max(0.005, xSpan * 0.035));

    // Only place tick labels in the active visible view
    const minXTick = Math.floor((range.xMin - xTickStep) / xTickStep);
    const maxXTick = Math.ceil((range.xMax + xTickStep) / xTickStep);
    for (let i = minXTick; i <= maxXTick; i++) {
        const x = Number((i * xTickStep).toFixed(6));
        if (Math.abs(x) > 0.0001 && Math.abs(x) <= limit) {
            gridGroup.add(createTextSprite(formatTickNumber(x), new THREE.Vector3(x, xLabelOffsetY, 0), labelFontSize));
        }
    }

    const minYTick = Math.floor((range.yMin - yTickStep) / yTickStep);
    const maxYTick = Math.ceil((range.yMax + yTickStep) / yTickStep);
    for (let i = minYTick; i <= maxYTick; i++) {
        const y = Number((i * yTickStep).toFixed(6));
        if (Math.abs(y) > 0.0001 && Math.abs(y) <= limit) {
            gridGroup.add(createTextSprite(formatTickNumber(y), new THREE.Vector3(yLabelOffsetX, y, 0), labelFontSize));
        }
    }

    const axisLabelSize = labelFontSize * 1.3;
    gridGroup.add(createTextSprite('x', new THREE.Vector3(range.xMax - xSpan * 0.04, ySpan * 0.03, 0), axisLabelSize));
    gridGroup.add(createTextSprite('y', new THREE.Vector3(xSpan * 0.03, range.yMax - ySpan * 0.04, 0), axisLabelSize));
    gridGroup.add(createTextSprite('0', new THREE.Vector3(yLabelOffsetX, xLabelOffsetY, 0), labelFontSize * 0.9));

    scene.add(gridGroup);
    return gridGroup;
};

const ORBIT_COLORS = {
    period1: { stable: '#27ae60', unstable: '#e74c3c', saddle: '#eedf32' },
    period2: { stable: '#27ae60', unstable: '#e74c3c', saddle: '#eedf32' },
    period3: { stable: '#27ae60', unstable: '#e74c3c', saddle: '#eedf32' },
    period4: { stable: '#27ae60', unstable: '#e74c3c', saddle: '#eedf32' },
    period5: { stable: '#27ae60', unstable: '#e74c3c', saddle: '#eedf32' },
    period6plus: { stable: '#27ae60', unstable: '#e74c3c', saddle: '#eedf32' },
    trajectory: '#ff00ff',  // Bright magenta for high visibility
    manifold: '#1e90ff',  // Blue for unstable manifold
    stableManifold: '#ffa500', // Orange for stable manifold
    attractor: '#27ae60',
    repeller: '#e74c3c',
    dualRepeller: '#800080', // Purple for dual repellers
    saddlePoint: '#eedf32',
    periodicBlue: '#3498db'
};

const SetValuedViz = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const sceneRef = useRef<THREE.Scene | null>(null);
    const cameraRef = useRef<THREE.OrthographicCamera | null>(null);
    const animationFrameRef = useRef<number | null>(null);
    const batchAnimationRef = useRef<number | null>(null);
    const viewTransitionFrameRef = useRef<number | null>(null);
    const manifoldDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    const readViewportSize = useCallback((): { width: number; height: number } => {
        const canvas = canvasRef.current;
        const bounds = canvas?.getBoundingClientRect();
        const parent = canvas?.parentElement;
        return {
            width: Math.max(1, bounds?.width || canvas?.clientWidth || parent?.clientWidth || window.innerWidth),
            height: Math.max(1, bounds?.height || canvas?.clientHeight || parent?.clientHeight || window.innerHeight),
        };
    }, []);

    const [dynamicSystem, setDynamicSystem] = useState<SystemId>('henon');
    const [customEquations, setCustomEquations] = useState<CustomEquations>(INITIAL_CUSTOM_EQUATIONS);
    const [draftCustomEquations, setDraftCustomEquations] = useState<CustomEquations>(INITIAL_CUSTOM_EQUATIONS);
    const [customParams, setCustomParams] = useState<CustomParameters>(INITIAL_CUSTOM_PARAMS);
    const [draftCustomParams, setDraftCustomParams] = useState<CustomParameters>(INITIAL_CUSTOM_PARAMS);
    const [equationError, setEquationError] = useState<string | null>(null);
    const [wasmModule, setWasmModule] = useState<BistWasmModule | null>(null);
    const [computeRequestId, setComputeRequestId] = useState(0);

    const [params, setParams] = useState<BistParameters>(INITIAL_PARAMS);
    const [draftParams, setDraftParams] = useState<BistParameters>(INITIAL_PARAMS);

    const [periodicState, setPeriodicState] = useState<PeriodicState>({
        orbits: [],
        isReady: false,
        showOrbits: false,
        computeMethod: null,
        resultRevision: 0,
        renderedRevision: 0,
        renderedPointCount: 0
    });
    const [periodicSearchSettings, setPeriodicSearchSettings] = useState<PeriodicSearchSettings>(DEFAULT_PERIODIC_SEARCH_SETTINGS);
    const [draftPeriodicSearchSettings, setDraftPeriodicSearchSettings] = useState<PeriodicSearchSettings>(DEFAULT_PERIODIC_SEARCH_SETTINGS);

    const [manifoldState, setManifoldState] = useState<ManifoldState>({
        manifolds: [],
        rawManifolds: [],
        stableManifolds: [],
        fixedPoints: [],
        intersections: [],
        isComputing: false,
        isReady: false,
        sourcePeriodicRevision: 0,
        showOrbits: true,
        showUnstableManifold: true,
        showDeterministicImageBoundary: false,
        showNoiseBalls: false,
        showBoundarySamplePoints: true,
        maximumManifoldPointSpacing: DEFAULT_MANIFOLD_SETTINGS.maximumPointSpacing,
        showStableManifold: true,
        intersectionThreshold: DEFAULT_MANIFOLD_SETTINGS.intersectionThreshold,
        currentPoint: { x: 0.1, y: 0.1, nx: 1.0, ny: 0.0 }, // 4D point for boundary map
        trajectoryPoints: [],
        iteration: 0,
        isRunning: false,
        hasStarted: false,
        showTrail: true,
        startPoint: { x: 0.1, y: 0.1, nx: 1.0, ny: 0.0 },
        selectedOrbitPeriod: 'all',
    });

    const [geometricOffsetState, setGeometricOffsetState] = useState<GeometricOffsetState>(() => {
        const contours = DEFAULT_GEOMETRIC_OFFSET_SETTINGS.contourEpsilons
            .map(createGeometricOffsetContour);
        return {
            editorMode: 'series',
            seriesStart: contours[0].epsilon,
            seriesEnd: contours[contours.length - 1].epsilon,
            seriesCount: contours.length,
            individualEpsilon: DEFAULT_GEOMETRIC_OFFSET_SETTINGS.contourEpsilon,
            contours,
            selectedContourId: contours[0].id,
            preimageSourceIds: [contours[0].id],
            inverseIterations: DEFAULT_GEOMETRIC_OFFSET_SETTINGS.inverseIterations,
            inverseDisplayMode: 'all',
            inverseColorMode: 'tracer',
            inverseColormap: 'rainbow',
            showInverseContours: true,
            isComputing: false,
            isComputingInverse: false,
            error: null,
            inverseError: null
        };
    });

    const [invariantSetState, setInvariantSetState] = useState<InvariantSetState>({
        seedMode: 'manual',
        boundaryPointCount: DEFAULT_INVARIANT_SET_SETTINGS.boundaryPointCount,
        forwardIterations: DEFAULT_INVARIANT_SET_SETTINGS.forwardIterations,
        showResult: true,
        isComputing: false,
        result: null,
        error: null,
    });

    const boundarySourceManifolds = manifoldState.manifolds;

    const verifiedBoundaryCycles = useMemo(
        () => buildVerifiedBoundaryCycles(boundarySourceManifolds),
        [boundarySourceManifolds]
    );

    const calculatedBoundaryBranches = useMemo(
        () => verifiedBoundaryCycles.length > 0
            ? verifiedBoundaryCycles
            : collectExtendedManifoldBranches(boundarySourceManifolds),
        [boundarySourceManifolds, verifiedBoundaryCycles]
    );

    const hasVerifiedBoundaryCycles = verifiedBoundaryCycles.length > 0;

    const boundaryLayers = useMemo(() => {
        if (calculatedBoundaryBranches.length === 0) {
            return {
                deterministicBranches: [] as ExtendedState[][],
                invariantBranches: [] as ExtendedState[][],
                noiseBallCenters: [] as ProjectedState[],
                unstableSampling: null,
                deterministicSampling: null,
                error: null as string | null,
            };
        }
        try {
            const deterministicBranches = calculatedBoundaryBranches
                .map(branch => reconstructDeterministicImageBoundary(branch, params.epsilon));
            const invariantBranches = deterministicBranches.map(branch => (
                branch.map(({ x, y, nx, ny }) => ({
                    x: x + params.epsilon * nx,
                    y: y + params.epsilon * ny,
                    nx,
                    ny,
                }))
            ));
            return {
                deterministicBranches,
                invariantBranches,
                noiseBallCenters: params.epsilon > 0 ? deterministicBranches.flat() : [],
                unstableSampling: summarizeOrderedBoundaryBranches(invariantBranches),
                deterministicSampling: summarizeOrderedBoundaryBranches(deterministicBranches),
                error: null,
            };
        } catch (error) {
            return {
                deterministicBranches: [] as ExtendedState[][],
                invariantBranches: [] as ExtendedState[][],
                noiseBallCenters: [] as ProjectedState[],
                unstableSampling: null,
                deterministicSampling: null,
                error: error instanceof Error ? error.message : String(error),
            };
        }
    }, [calculatedBoundaryBranches, params.epsilon]);

    const hasBoundarySamples = calculatedBoundaryBranches.length > 0
        && boundaryLayers.error === null
        && boundaryLayers.deterministicBranches.length > 0;

    const [filters, setFilters] = useState<OrbitFilters>({
        period1: true, period2: true, period3: true,
        period4: true, period5: true, period6plus: true
    });

    const isCustomSystem = isCustomSystemId(dynamicSystem);
    const type = systemTypeFor(dynamicSystem);
    const supportsPeriodicSearchSettings = supportsPeriodicSearch(dynamicSystem);
    const activeCustomKey = isCustomSystem ? dynamicSystem : 'custom';
    const activeAppliedCustomEquations = isCustomSystem ? customEquations[activeCustomKey] : customEquations.custom;
    const activeDraftCustomEquations = isCustomSystem ? draftCustomEquations[activeCustomKey] : draftCustomEquations.custom;
    const activeAppliedCustomParams = isCustomSystem ? customParams[activeCustomKey] : EMPTY_ARRAY;
    const activeDraftCustomParams = isCustomSystem ? draftCustomParams[activeCustomKey] : EMPTY_ARRAY;
    const appliedParamValidation = useMemo(() => {
        if (!isCustomSystem) {
            return DEFAULT_VALIDATION;
        }
        return normalizeParams(activeAppliedCustomParams);
    }, [isCustomSystem, activeAppliedCustomParams]);
    const draftParamValidation = useMemo(() => {
        if (!isCustomSystem) {
            return DEFAULT_VALIDATION;
        }
        return normalizeParams(activeDraftCustomParams);
    }, [isCustomSystem, activeDraftCustomParams]);
    const hasPendingInputChanges = useMemo(() => {
        const paramsDirty = JSON.stringify(draftParams) !== JSON.stringify(params);
        const periodicSearchDirty = supportsPeriodicSearchSettings && (
            draftPeriodicSearchSettings.gridSize !== periodicSearchSettings.gridSize
            || draftPeriodicSearchSettings.thetaGridSize !== periodicSearchSettings.thetaGridSize
            || draftPeriodicSearchSettings.residualThreshold !== periodicSearchSettings.residualThreshold
            || draftPeriodicSearchSettings.useContinuation !== periodicSearchSettings.useContinuation
        );
        if (!isCustomSystem) {
            return paramsDirty || periodicSearchDirty;
        }
        const equationsDirty = JSON.stringify(activeDraftCustomEquations) !== JSON.stringify(activeAppliedCustomEquations);
        const customParamsDirty = JSON.stringify(activeDraftCustomParams) !== JSON.stringify(activeAppliedCustomParams);
        return paramsDirty || equationsDirty || customParamsDirty || periodicSearchDirty;
    }, [
        draftParams,
        params,
        isCustomSystem,
        supportsPeriodicSearchSettings,
        draftPeriodicSearchSettings,
        periodicSearchSettings,
        activeDraftCustomEquations,
        activeAppliedCustomEquations,
        activeDraftCustomParams,
        activeAppliedCustomParams
    ]);

    // Parameter sweep state
    const [sweepState, setSweepState] = useState<SweepState>({
        results: null,
        isComputing: false,
        error: null,
        sweepParam: 'a',
        sweepMin: 0.1,
        sweepMax: 2.0,
        numSamples: 10,
        maxPeriod: 3,
    });

    // BDE Simulator
    const [bdeState, setBdeState] = useState<BdeState>({
        points: [],
        isRunning: false
    });
    const bdeSimRef = useRef<BdeSimulator | null>(null);
    const bdeAnimRef = useRef<number | null>(null);
    const postApplyActionRef = useRef<'step' | 'play' | null>(null);

    // Ulam method state
    const [ulamState, setUlamState] = useState<UlamState>({
        gridBoxes: [],
        invariantMeasure: null,
        leftEigenvector: null, // backward invariant measure
        transitions: null, // array of {index, probability}
        selectedBoxIndex: null,
        currentBoxIndex: -1,
        isComputing: false,
        subdivisions: DEFAULT_ULAM_SETTINGS.subdivisions,
        pointsPerBox: DEFAULT_ULAM_SETTINGS.pointsPerBox, // 8x8 grid per box
        epsilon: DEFAULT_ULAM_SETTINGS.epsilon, // epsilon ball radius for set-valued transitions
        showUlamOverlay: false,
        showTransitions: true,
        showCurrentBox: true, // highlight box containing current trajectory point
        needsRecompute: false // flag for auto-recompute
    });

    // Parameter animation state for manifold mode
    const [animationState, setAnimationState] = useState<AnimationState>({
        isAnimating: false,
        isPreparing: false,
        parameter: 'a', // 'a', 'b', or 'epsilon'
        rangeValue: 0.1, // the range amount (e.g., 0.1 means go from current to current+0.1 or current-0.1)
        direction: 1, // +1 for positive direction, -1 for negative direction
        steps: 10, // number of steps to divide the range
        currentStep: 0, // current step in the animation
        baseValue: null, // the original value when animation started
        targetValue: null, // the target value at the end of animation
        awaitingResult: false,
        expectedPeriodicRevision: null
    });

    // Video recording state
    const [recordingState, setRecordingState] = useState<RecordingState>({
        isRecording: false,
        isEncoding: false,
        frameCount: 0,
        recordingEnabled: false, // toggle for recording with animation
        encodingProgress: 0,
        error: null
    });
    const [experimentStatus, setExperimentStatus] = useState<ExperimentStatus | null>(null);

    const recordedFramesRef = useRef<ImageBitmap[]>([]);
    const encoderWorkerRef = useRef<Worker | null>(null);
    const animationSearchSettingsRef = useRef<PeriodicSearchSettings | null>(null);
    const pendingAnimationStartRef = useRef(false);

    const periodicComputationRevisionRef = useRef(0);
    const geometricOffsetRequestIdRef = useRef(0);
    const inverseOffsetRequestIdRef = useRef(0);
    const invariantSetRequestIdRef = useRef(0);
    const ulamDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const ulamSupportRef = useRef<SupportGrid | null>(null);
    const ulamTransitionsRequestRef = useRef(0);

    const [tooltip, setTooltip] = useState<TooltipState>({
        visible: false,
        x: 0,
        y: 0,
        data: null
    });
    const [viewRange, setViewRange] = useState<ViewRange>(DEFAULT_VIEW_RANGE);
    const computationViewRangeRef = useRef(viewRange);
    const [viewportRange, setViewportRange] = useState<ViewRange>(DEFAULT_VIEWPORT_RANGE);
    const viewportRangeRef = useRef(viewportRange);
    const viewportRangeTargetRef = useRef(viewportRange);
    const gridGroupRef = useRef<THREE.Group | null>(null);

    const [isPanMode, setIsPanMode] = useState(false);
    const isDraggingRef = useRef(false);
    const hasDraggedRef = useRef(false);
    const dragStartRef = useRef<{ x: number; y: number; range: ViewRange } | null>(null);

    const handlePanMode = useCallback(() => {
        setIsPanMode(prev => !prev);
    }, []);

    const raycasterRef = useRef(new THREE.Raycaster());
    const mouseRef = useRef(new THREE.Vector2());

    const runComputeTask = useComputeWorker();

    const updatePeriodicSearchSettings = useCallback((patch: Partial<PeriodicSearchSettings>) => {
        setDraftPeriodicSearchSettings(prev => normalizePeriodicSearchSettings({ ...prev, ...patch }, prev));
    }, []);


    const applyViewRangeToCamera = useCallback((range: ViewRange) => {
        const camera = cameraRef.current;
        if (!camera) return;

        const { width, height } = readViewportSize();
        const { width: frustumWidth, height: frustumHeight } = orthographicFrustumSize(range, width, height);

        camera.left = -frustumWidth / 2;
        camera.right = frustumWidth / 2;
        camera.top = frustumHeight / 2;
        camera.bottom = -frustumHeight / 2;

        const centerX = (range.xMin + range.xMax) / 2;
        const centerY = (range.yMin + range.yMax) / 2;
        camera.position.set(centerX, centerY, 5);
        camera.lookAt(centerX, centerY, 0);
        camera.updateProjectionMatrix();
    }, [readViewportSize]);

    const cancelViewRangeTransition = useCallback(() => {
        if (viewTransitionFrameRef.current !== null) {
            cancelAnimationFrame(viewTransitionFrameRef.current);
            viewTransitionFrameRef.current = null;
        }
    }, []);

    const transitionViewRange = useCallback((targetRange: ViewRange) => {
        const target = constrainViewRange(targetRange, computationViewRangeRef.current);
        const start = { ...viewportRangeRef.current };
        cancelViewRangeTransition();
        viewportRangeTargetRef.current = target;

        const reduceMotion = typeof window.matchMedia === 'function'
            && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (!cameraRef.current || reduceMotion) {
            viewportRangeRef.current = target;
            setViewportRange(target);
            return;
        }

        const scene = sceneRef.current;
        if (scene) {
            if (gridGroupRef.current) {
                scene.remove(gridGroupRef.current);
            }
            gridGroupRef.current = createCoordinateSystem(scene, target);
        }

        const startedAt = performance.now();
        const durationMs = 180;
        const animateRange = (timestamp: number): void => {
            const progress = Math.min(1, (timestamp - startedAt) / durationMs);
            const eased = 1 - ((1 - progress) ** 3);
            const intermediate = {
                xMin: start.xMin + (target.xMin - start.xMin) * eased,
                xMax: start.xMax + (target.xMax - start.xMax) * eased,
                yMin: start.yMin + (target.yMin - start.yMin) * eased,
                yMax: start.yMax + (target.yMax - start.yMax) * eased
            };
            viewportRangeRef.current = intermediate;
            applyViewRangeToCamera(intermediate);

            if (progress < 1) {
                viewTransitionFrameRef.current = requestAnimationFrame(animateRange);
                return;
            }

            viewTransitionFrameRef.current = null;
            viewportRangeRef.current = target;
            setViewportRange(target);
        };

        viewTransitionFrameRef.current = requestAnimationFrame(animateRange);
    }, [applyViewRangeToCamera, cancelViewRangeTransition]);

    const updateViewRange = useCallback((patch: Partial<ViewRange>) => {
        cancelViewRangeTransition();
        const next = { ...computationViewRangeRef.current };
        Object.entries(patch).forEach(([key, value]) => {
            if (Number.isFinite(value)) {
                next[key as keyof ViewRange] = value as number;
            }
        });
        const normalized = normalizeViewRange(next);
        computationViewRangeRef.current = normalized;
        viewportRangeRef.current = normalized;
        viewportRangeTargetRef.current = normalized;
        setViewRange(normalized);
        setViewportRange(normalized);
    }, [cancelViewRangeTransition]);

    const resetViewRange = useCallback(() => {
        cancelViewRangeTransition();
        computationViewRangeRef.current = DEFAULT_VIEW_RANGE;
        viewportRangeRef.current = DEFAULT_VIEWPORT_RANGE;
        viewportRangeTargetRef.current = DEFAULT_VIEWPORT_RANGE;
        setViewRange(DEFAULT_VIEW_RANGE);
        setViewportRange(DEFAULT_VIEWPORT_RANGE);
    }, [cancelViewRangeTransition]);

    const resetViewportRange = useCallback(() => {
        transitionViewRange(computationViewRangeRef.current);
    }, [transitionViewRange]);

    const handleZoomIn = useCallback(() => {
        const baseRange = viewTransitionFrameRef.current !== null
            ? viewportRangeTargetRef.current
            : viewportRangeRef.current;
        transitionViewRange(zoomViewRange(baseRange, ZOOM_IN_FACTOR));
    }, [transitionViewRange]);

    const handleZoomOut = useCallback(() => {
        const baseRange = viewTransitionFrameRef.current !== null
            ? viewportRangeTargetRef.current
            : viewportRangeRef.current;
        transitionViewRange(zoomViewRange(baseRange, ZOOM_OUT_FACTOR));
    }, [transitionViewRange]);

    useEffect(() => {
        if (!canvasRef.current) return;

        const scene = new THREE.Scene();
        sceneRef.current = scene;

        const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 1000);
        camera.position.z = 5;
        cameraRef.current = camera;

        const renderer = new THREE.WebGLRenderer({
            canvas: canvasRef.current,
            antialias: true,
            alpha: true,
            preserveDrawingBuffer: true
        });
        const initialSize = readViewportSize();
        renderer.setSize(initialSize.width, initialSize.height, false);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        rendererRef.current = renderer;

        const initialViewportRange = viewportRangeRef.current;
        applyViewRangeToCamera(initialViewportRange);
        gridGroupRef.current = createCoordinateSystem(scene, initialViewportRange);

        const handleResize = () => {
            const range = viewportRangeRef.current;
            const size = readViewportSize();
            applyViewRangeToCamera(range);
            renderer.setSize(size.width, size.height, false);
            if (gridGroupRef.current) {
                scene.remove(gridGroupRef.current);
            }
            gridGroupRef.current = createCoordinateSystem(scene, range);
            scene.traverse(object => {
                if (object instanceof Line2) {
                    object.material.resolution.set(
                        size.width,
                        size.height
                    );
                }
            });
        };
        const resizeObserver = typeof ResizeObserver === 'undefined'
            ? null
            : new ResizeObserver(handleResize);
        resizeObserver?.observe(canvasRef.current);
        window.addEventListener('resize', handleResize);

        const animate = () => {
            animationFrameRef.current = requestAnimationFrame(animate);
            renderer.render(scene, camera);
        };
        animate();

        return () => {
            window.removeEventListener('resize', handleResize);
            resizeObserver?.disconnect();
            if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
            if (batchAnimationRef.current) cancelAnimationFrame(batchAnimationRef.current);
            if (viewTransitionFrameRef.current) cancelAnimationFrame(viewTransitionFrameRef.current);
            renderer.dispose();
        };
    }, [applyViewRangeToCamera, readViewportSize]);

    useEffect(() => {
        viewportRangeRef.current = viewportRange;
        viewportRangeTargetRef.current = viewportRange;
        const scene = sceneRef.current;
        if (!scene) return;
        if (gridGroupRef.current) {
            scene.remove(gridGroupRef.current);
        }
        gridGroupRef.current = createCoordinateSystem(scene, viewportRange);
        applyViewRangeToCamera(viewportRange);
    }, [viewportRange, applyViewRangeToCamera, readViewportSize]);

    useEffect(() => {
        computationViewRangeRef.current = viewRange;
    }, [viewRange]);

    useEffect(() => {
        const loadWasm = async () => {
            try {
                const wasm = await import('../pkg/bist');
                await wasm.default();
                setWasmModule(wasm);
                console.log('WASM module loaded successfully');
            } catch (err) {
                console.error('Failed to load WASM module:', err);
            }
        };
        loadWasm();
    }, []);

    const validateCustomDraft = useCallback((
        candidateEquations: CustomEquation,
        candidateParams: CustomParameter[],
        candidateEpsilon: number,
    ): { valid: boolean; error?: string } => {
        if (!isCustomSystem) {
            return { valid: true };
        }
        if (!draftParamValidation.valid) {
            const firstError = draftParamValidation.errors.find(Boolean);
            return { valid: false, error: firstError ? `Parameter error: ${firstError}` : 'Invalid parameters' };
        }
        if (!wasmModule) {
            return { valid: true };
        }
        try {
            if (dynamicSystem === 'custom') {
                const result = wasmModule.evaluate_user_defined_map(
                    0.5, 0.5,
                    candidateEquations.xEq, candidateEquations.yEq,
                    candidateParams,
                    candidateEpsilon
                );
                if (result && isFinite(result.x) && isFinite(result.y)) {
                    return { valid: true };
                }
            } else {
                const result = wasmModule.evaluate_user_defined_ode(
                    0.5, 0.5,
                    candidateEquations.xEq, candidateEquations.yEq,
                    candidateParams
                );
                if (result && isFinite(result.x) && isFinite(result.y)) {
                    return { valid: true };
                }
            }
            return { valid: false, error: 'Equations produce non-finite values' };
        } catch (err) {
            return { valid: false, error: String(err).replace('Error: ', '') };
        }
    }, [draftParamValidation, dynamicSystem, isCustomSystem, wasmModule]);

    const applyInputsAndRecomputeWithSearchSettings = useCallback((
        nextPeriodicSearchSettings: PeriodicSearchSettings,
    ) => {
        const nextDraftParams = {
            ...draftParams,
            a: clampToRange(draftParams.a, -PARAM_ABS_LIMIT, PARAM_ABS_LIMIT, params.a),
            b: clampToRange(draftParams.b, -PARAM_ABS_LIMIT, PARAM_ABS_LIMIT, params.b)
        };
        if (nextDraftParams.a !== draftParams.a || nextDraftParams.b !== draftParams.b) {
            setDraftParams(prev => ({
                ...prev,
                a: nextDraftParams.a,
                b: nextDraftParams.b
            }));
        }
        if (isCustomSystem) {
            const nextDraftEquations = draftCustomEquations[activeCustomKey];
            const nextDraftCustomParams = draftParamValidation.normalized;
            const validation = validateCustomDraft(nextDraftEquations, nextDraftCustomParams, nextDraftParams.epsilon);
            if (!validation.valid) {
                setEquationError(validation.error ?? 'Invalid custom equations.');
                return;
            }
            setCustomEquations(prev => ({
                ...prev,
                [activeCustomKey]: {
                    xEq: nextDraftEquations.xEq,
                    yEq: nextDraftEquations.yEq
                }
            }));
            setCustomParams(prev => ({
                ...prev,
                [activeCustomKey]: nextDraftCustomParams
            }));
        }
        setEquationError(null);
        setParams(nextDraftParams);
        setPeriodicSearchSettings(nextPeriodicSearchSettings);
        setComputeRequestId(prev => prev + 1);
        return true;
    }, [draftParams, params.a, params.b, isCustomSystem, draftCustomEquations, activeCustomKey, draftParamValidation, validateCustomDraft]);

    const applyInputsAndRecompute = useCallback(() => (
        applyInputsAndRecomputeWithSearchSettings(draftPeriodicSearchSettings)
    ), [applyInputsAndRecomputeWithSearchSettings, draftPeriodicSearchSettings]);

    const runPeriodicGridSearch = useCallback(() => {
        const gridSearchSettings = forceFullGridSearchSettings(draftPeriodicSearchSettings);
        setDraftPeriodicSearchSettings(gridSearchSettings);
        return applyInputsAndRecomputeWithSearchSettings(gridSearchSettings);
    }, [applyInputsAndRecomputeWithSearchSettings, draftPeriodicSearchSettings]);

    const applyIfNeededBeforeAction = useCallback((action: 'step' | 'play'): boolean => {
        if (!hasPendingInputChanges) {
            return false;
        }
        postApplyActionRef.current = action;
        const applied = applyInputsAndRecompute();
        if (!applied) {
            postApplyActionRef.current = null;
        }
        return Boolean(applied);
    }, [hasPendingInputChanges, applyInputsAndRecompute]);

    useEffect(() => {
        if (!equationError) return;
        setEquationError(null);
    }, [draftCustomEquations, draftCustomParams, draftParams.epsilon, dynamicSystem, equationError]);

    const systemLabel = useMemo(() => {
        const labels = { henon: 'Hénon', duffing: 'Duffing Map', duffing_ode: 'Duffing ODE', custom: 'Custom', custom_ode: 'Custom ODE' };
        return labels[dynamicSystem] || dynamicSystem;
    }, [dynamicSystem]);

    const paramOverlayText = useMemo(() => {
        if (dynamicSystem === 'duffing_ode') {
            return `δ = ${(params.delta || 0).toFixed(4)}  h = ${(params.h || 0).toFixed(4)}  ε = ${(params.epsilon || 0).toFixed(4)}`;
        } else if (dynamicSystem === 'custom_ode') {
            const cp = (customParams.custom_ode || []).map(p => `${p.name} = ${p.value.toFixed(4)}`).join('  ');
            return cp || `h = ${(params.h || 0).toFixed(4)}  ε = ${(params.epsilon || 0).toFixed(4)}`;
        } else if (dynamicSystem === 'custom') {
            const cp = (customParams.custom || []).map(p => `${p.name} = ${p.value.toFixed(4)}`).join('  ');
            return cp || `ε = ${(params.epsilon || 0).toFixed(4)}`;
        }
        return `a = ${(params.a || 0).toFixed(4)}  b = ${(params.b || 0).toFixed(4)}  ε = ${(params.epsilon || 0).toFixed(4)}`;
    }, [dynamicSystem, params, customParams]);

    const systemFilePrefix = useMemo(() => {
        const prefixes = { henon: 'henon', duffing: 'duffing_map', duffing_ode: 'duffing_ode', custom: 'custom', custom_ode: 'custom_ode' };
        return prefixes[dynamicSystem] || 'system';
    }, [dynamicSystem]);

    const paramFileString = useMemo(() => {
        if (dynamicSystem === 'duffing_ode') {
            const dStr = (params.delta || 0).toFixed(3).replace('.', 'p').replace('-', 'm');
            const hStr = (params.h || 0).toFixed(3).replace('.', 'p').replace('-', 'm');
            const epsStr = (params.epsilon || 0).toFixed(4).replace('.', 'p').replace('-', 'm');
            return `d${dStr}_h${hStr}_eps${epsStr}`;
        }
        const aStr = (params.a || 0).toFixed(3).replace('.', 'p').replace('-', 'm');
        const bStr = (params.b || 0).toFixed(3).replace('.', 'p').replace('-', 'm');
        const epsStr = (params.epsilon || 0).toFixed(4).replace('.', 'p').replace('-', 'm');
        return `a${aStr}_b${bStr}_eps${epsStr}`;
    }, [dynamicSystem, params]);

    useEffect(() => {
        let paramPatch = null;
        if (dynamicSystem === 'duffing') {
            paramPatch = { a: 2.75, b: 0.2 };
        } else if (dynamicSystem === 'duffing_ode') {
            paramPatch = { delta: 0.15, h: 0.05, epsilon: 0.1 };
        } else if (dynamicSystem === 'custom_ode') {
            paramPatch = { h: 0.05, epsilon: 0.1 };
        } else if (dynamicSystem === 'custom') {
            paramPatch = { maxPeriod: 3 };
        } else {
            paramPatch = { a: 0.4, b: 0.3 };
        }
        if (paramPatch) {
            setParams(prev => ({ ...prev, ...paramPatch }));
            setDraftParams(prev => ({ ...prev, ...paramPatch }));
        }

        setManifoldState(prev => ({
            ...prev,
            isRunning: false,
            hasStarted: false,
            iteration: 0,
            trajectoryPoints: [],
            manifolds: [],
            rawManifolds: [],
            stableManifolds: [],
            fixedPoints: [],
            intersections: [],
            showUnstableManifold: false,
            showDeterministicImageBoundary: false,
            showNoiseBalls: false,
            showBoundarySamplePoints: true,
            maximumManifoldPointSpacing: DEFAULT_MANIFOLD_SETTINGS.maximumPointSpacing,
            showStableManifold: false,
            showOrbits: true,
            showTrail: true,
        }));
        setBdeState(prev => ({
            ...prev,
            isRunning: false,
            points: []
        }));
        setPeriodicState(prev => ({
            ...prev,
            orbits: [],
            showOrbits: false,
        }));
        setUlamState(prev => ({
            ...prev,
            showUlamOverlay: false,
            gridBoxes: [],
            invariantMeasure: null,
            transitions: null,
            currentBoxIndex: -1,
            selectedBoxIndex: null,
        }));
        setSweepState(prev => ({
            ...prev,
            results: null,
            error: null,
        }));
    }, [dynamicSystem]);

    const computeJacobian = useCallback((x: number, y: number) => {
        if (dynamicSystem === 'duffing_ode') {
            const h = params.h;
            const delta = params.delta;
            // Df = [[0, 1], [1 - 3*x^2, -delta]]
            // DF_h = I + h * Df
            const j11 = 1.0;
            const j12 = h;
            const j21 = h * (1.0 - 3.0 * x * x);
            const j22 = 1.0 - h * delta;
            const trace = j11 + j22;
            const det = j11 * j22 - j12 * j21;
            return { j11, j12, j21, j22, trace, det };
        }

        if (dynamicSystem === 'custom' && wasmModule && appliedParamValidation.valid) {
            const h = 1e-5;
            const evalMap = (xv: number, yv: number) => wasmModule.evaluate_user_defined_map(
                xv, yv,
                activeAppliedCustomEquations.xEq, activeAppliedCustomEquations.yEq,
                appliedParamValidation.normalized,
                params.epsilon
            );
            const f1 = evalMap(x + h, y);
            const f2 = evalMap(x - h, y);
            const f3 = evalMap(x, y + h);
            const f4 = evalMap(x, y - h);
            if (!f1 || !f2 || !f3 || !f4) {
                return { j11: 0, j12: 0, j21: 0, j22: 0, trace: 0, det: 0 };
            }
            const j11 = (f1.x - f2.x) / (2 * h);
            const j12 = (f3.x - f4.x) / (2 * h);
            const j21 = (f1.y - f2.y) / (2 * h);
            const j22 = (f3.y - f4.y) / (2 * h);
            const trace = j11 + j22;
            const det = j11 * j22 - j12 * j21;
            return { j11, j12, j21, j22, trace, det };
        }

        if (dynamicSystem === 'custom_ode' && wasmModule && appliedParamValidation.valid) {
            const h = 1e-5;
            const evalVF = (xv: number, yv: number) => wasmModule.evaluate_user_defined_ode(
                xv, yv,
                activeAppliedCustomEquations.xEq, activeAppliedCustomEquations.yEq,
                appliedParamValidation.normalized
            );
            const f1 = evalVF(x + h, y);
            const f2 = evalVF(x - h, y);
            const f3 = evalVF(x, y + h);
            const f4 = evalVF(x, y - h);
            if (!f1 || !f2 || !f3 || !f4) {
                return { j11: 0, j12: 0, j21: 0, j22: 0, trace: 0, det: 0 };
            }
            const dfx_dx = (f1.x - f2.x) / (2 * h);
            const dfx_dy = (f3.x - f4.x) / (2 * h);
            const dfy_dx = (f1.y - f2.y) / (2 * h);
            const dfy_dy = (f3.y - f4.y) / (2 * h);
            const step = params.h;
            const j11 = 1.0 + step * dfx_dx;
            const j12 = step * dfx_dy;
            const j21 = step * dfy_dx;
            const j22 = 1.0 + step * dfy_dy;
            const trace = j11 + j22;
            const det = j11 * j22 - j12 * j21;
            return { j11, j12, j21, j22, trace, det };
        }

        const a = params.a;
        const b = params.b;
        const j11 = -2 * a * x;
        const j12 = 1;
        const j21 = b;
        const j22 = 0;
        const trace = j11 + j22;
        const det = j11 * j22 - j12 * j21;
        return { j11, j12, j21, j22, trace, det };
    }, [params.a, params.b, params.delta, params.h, params.epsilon, dynamicSystem, activeAppliedCustomEquations, appliedParamValidation, wasmModule]);

    const handleMouseMove = useCallback((event: MouseEvent) => {
        if (!canvasRef.current || !sceneRef.current || !cameraRef.current) return;

        const rect = canvasRef.current.getBoundingClientRect();
        mouseRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouseRef.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);

        if (ulamState.showUlamOverlay && ulamState.gridBoxes.length > 0) {
            const ulamMesh = sceneRef.current.getObjectByName('ulam-grid');
            if (ulamMesh) {
                const intersects = raycasterRef.current.intersectObject(ulamMesh);
                if (intersects.length > 0 && intersects[0].instanceId !== undefined) {
                    const boxIndex = intersects[0].instanceId;
                    const box = ulamState.gridBoxes[boxIndex];
                    const measure = ulamState.invariantMeasure ? ulamState.invariantMeasure[boxIndex] : 0;
                    const maxMeasure = ulamState.invariantMeasure ? Math.max(...ulamState.invariantMeasure) : 1;

                    let numTransitions = 0;
                    let topTransitions: UlamTransition[] = [];
                    if (ulamState.selectedBoxIndex === boxIndex && Array.isArray(ulamState.transitions)) {
                        numTransitions = ulamState.transitions.length;
                        topTransitions = [...ulamState.transitions]
                            .sort((a, b) => (b.probability || 0) - (a.probability || 0))
                            .slice(0, 3);
                    }

                    setTooltip({
                        visible: true,
                        x: event.clientX,
                        y: event.clientY,
                        data: {
                            type: 'Ulam Box',
                            boxIndex: boxIndex,
                            pos: { x: box.center[0], y: box.center[1] },
                            measure: measure,
                            measurePercent: maxMeasure > 0 ? (measure / maxMeasure * 100) : 0,
                            numTransitions: numTransitions,
                            topTransitions: topTransitions,
                            isCurrentBox: boxIndex === ulamState.currentBoxIndex
                        }
                    });
                    return;
                }
            }
        }

        const meshes: THREE.Mesh[] = [];
        sceneRef.current.traverse((obj) => {
            if (obj instanceof THREE.Mesh && (
                obj.userData.type === 'orbit'
                || obj.userData.type === 'fixedPoint'
            )) {
                meshes.push(obj);
            }
        });

        const intersects = raycasterRef.current.intersectObjects(meshes, false);

        if (intersects.length > 0) {
            const hit = intersects[0].object;
            const data = hit.userData;
            const jac = computeJacobian(data.pos.x, data.pos.y);

            setTooltip({
                visible: true,
                x: event.clientX,
                y: event.clientY,
                data: {
                    type: data.type === 'fixedPoint' ? 'Fixed Point' : 'Periodic Point',
                    period: data.period,
                    stability: data.stability,
                    pos: data.pos,
                    normal: data.normal,
                    eigenvalues: data.eigenvalues,
                    jacobian: jac,
                    orbitSize: data.orbitPoints?.length || 1
                }
            });
        } else {
            setTooltip(prev => prev.visible ? { ...prev, visible: false } : prev);
        }
    }, [computeJacobian, ulamState.showUlamOverlay, ulamState.gridBoxes, ulamState.invariantMeasure, ulamState.currentBoxIndex, ulamState.selectedBoxIndex, ulamState.transitions]);

    const requestUlamTransitions = useCallback((
        index: number,
        mode: 'selected' | 'current' = 'selected',
    ) => {
        if (index < 0) return;
        const requestId = ++ulamTransitionsRequestRef.current;
        runComputeTask('getUlamTransitions', { index }).then((transitions) => {
            if (requestId !== ulamTransitionsRequestRef.current) return;
            setUlamState(prev => {
                if (mode === 'selected' && prev.selectedBoxIndex !== index) return prev;
                if (mode === 'current' && (prev.currentBoxIndex !== index || !prev.showCurrentBox)) return prev;
                return { ...prev, transitions: transitions || [] };
            });
        }).catch((err) => {
            console.warn('Failed to fetch Ulam transitions:', err);
        });
    }, [runComputeTask]);

    const handleUlamClick = useCallback((index: number) => {
        setUlamState(prev => ({
            ...prev,
            selectedBoxIndex: index,
            transitions: null
        }));
        requestUlamTransitions(index, 'selected');
    }, [requestUlamTransitions]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const handlePointerDown = (event: PointerEvent): void => {
            if (event.button !== 0) return;
            try {
                canvas.setPointerCapture(event.pointerId);
            } catch {
                // Ignore if pointer capture fails
            }
            isDraggingRef.current = true;
            hasDraggedRef.current = false;
            dragStartRef.current = {
                x: event.clientX,
                y: event.clientY,
                range: { ...viewportRangeRef.current },
            };
            canvas.classList.add('panning');
        };

        const handlePointerMove = (event: PointerEvent): void => {
            handleMouseMove(event);

            if (!isDraggingRef.current || !dragStartRef.current) return;

            const dx = event.clientX - dragStartRef.current.x;
            const dy = event.clientY - dragStartRef.current.y;
            const distance = Math.hypot(dx, dy);

            if (distance > 3) {
                hasDraggedRef.current = true;
            }

            if (hasDraggedRef.current) {
                const rect = canvas.getBoundingClientRect();
                const startRange = dragStartRef.current.range;

                const camera = cameraRef.current;
                if (!camera) return;

                const frustumWidth = camera.right - camera.left;
                const frustumHeight = camera.top - camera.bottom;

                const worldDx = (dx / (rect.width || 1)) * frustumWidth;
                const worldDy = (dy / (rect.height || 1)) * frustumHeight;

                const newRange = constrainViewRange({
                    xMin: startRange.xMin - worldDx,
                    xMax: startRange.xMax - worldDx,
                    yMin: startRange.yMin + worldDy,
                    yMax: startRange.yMax + worldDy,
                }, computationViewRangeRef.current);

                cancelViewRangeTransition();
                viewportRangeRef.current = newRange;
                viewportRangeTargetRef.current = newRange;
                applyViewRangeToCamera(newRange);
            }
        };

        const handlePointerUp = (event: PointerEvent): void => {
            if (isDraggingRef.current) {
                try {
                    canvas.releasePointerCapture(event.pointerId);
                } catch {
                    // Ignore pointer release error
                }
                canvas.classList.remove('panning');
            }
            const wasDragging = hasDraggedRef.current;
            isDraggingRef.current = false;
            dragStartRef.current = null;

            if (wasDragging) {
                setViewportRange({ ...viewportRangeRef.current });
            }
        };

        const handleClick = (event: MouseEvent): void => {
            if (hasDraggedRef.current) {
                hasDraggedRef.current = false;
                return;
            }

            const camera = cameraRef.current;
            const scene = sceneRef.current;
            if (!camera || !scene) return;
            const rect = canvas.getBoundingClientRect();
            const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            if (ulamState.showUlamOverlay && ulamState.gridBoxes.length) {
                raycasterRef.current.setFromCamera(new THREE.Vector2(x, y), camera);
                const ulamMesh = scene.getObjectByName('ulam-grid');

                if (ulamMesh) {
                    const intersects = raycasterRef.current.intersectObject(ulamMesh);
                    if (intersects.length > 0) {
                        const instanceId = intersects[0].instanceId;
                        if (instanceId !== undefined) {
                            handleUlamClick(instanceId);
                            return;
                        }
                    } else {
                        setUlamState(prev => ({ ...prev, selectedBoxIndex: null, transitions: null }));
                        return;
                    }
                }
            }

            raycasterRef.current.setFromCamera(new THREE.Vector2(x, y), camera);
            const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
            const target = new THREE.Vector3();
            raycasterRef.current.ray.intersectPlane(plane, target);
            if (target) {
                setManifoldState(prev => {
                    const newStart = { ...prev.startPoint, x: target.x, y: target.y };
                    return applyStartPointUpdate(prev, newStart);
                });
            }
        };

        const handleWheel = (event: WheelEvent): void => {
            event.preventDefault();
            const factor = event.deltaY > 0 ? ZOOM_OUT_FACTOR : ZOOM_IN_FACTOR;
            const baseRange = viewTransitionFrameRef.current !== null
                ? viewportRangeTargetRef.current
                : viewportRangeRef.current;
            transitionViewRange(zoomViewRange(baseRange, factor));
        };

        canvas.addEventListener('pointerdown', handlePointerDown);
        canvas.addEventListener('pointermove', handlePointerMove);
        canvas.addEventListener('pointerup', handlePointerUp);
        canvas.addEventListener('pointercancel', handlePointerUp);
        canvas.addEventListener('click', handleClick);
        canvas.addEventListener('wheel', handleWheel, { passive: false });

        return () => {
            canvas.removeEventListener('pointerdown', handlePointerDown);
            canvas.removeEventListener('pointermove', handlePointerMove);
            canvas.removeEventListener('pointerup', handlePointerUp);
            canvas.removeEventListener('pointercancel', handlePointerUp);
            canvas.removeEventListener('click', handleClick);
            canvas.removeEventListener('wheel', handleWheel);
        };
    }, [handleMouseMove, ulamState.showUlamOverlay, ulamState.gridBoxes.length, handleUlamClick, applyViewRangeToCamera, cancelViewRangeTransition, readViewportSize, transitionViewRange]);

    useEffect(() => {
        if (!wasmModule) return;

        let cancelled = false;
        const resultRevision = ++periodicComputationRevisionRef.current;
        const requestSearchSettings = animationSearchSettingsRef.current || {
            gridSize: periodicSearchSettings.gridSize,
            thetaGridSize: periodicSearchSettings.thetaGridSize,
            residualThreshold: periodicSearchSettings.residualThreshold,
            useContinuation: periodicSearchSettings.useContinuation
        };
        if (dynamicSystem === 'custom_ode' || (dynamicSystem === 'custom' && !appliedParamValidation.valid)) {
            ulamSupportRef.current = null;
            setPeriodicState(prev => ({
                ...prev,
                isReady: true,
                orbits: [],
                computeMethod: null,
                resultRevision
            }));
            return;
        }
        setPeriodicState(beginPeriodicRefresh);

        const initSystem = async () => {
            try {
                if (cancelled) return;
                const result = await runComputeTask('computePeriodic', {
                    dynamicSystem,
                    params: {
                        a: params.a,
                        b: params.b,
                        delta: params.delta,
                        h: params.h,
                        epsilon: params.epsilon,
                        maxPeriod: params.maxPeriod
                    },
                    viewRange: {
                        xMin: viewRange.xMin,
                        xMax: viewRange.xMax,
                        yMin: viewRange.yMin,
                        yMax: viewRange.yMax
                    },
                    periodicSearchSettings: {
                        gridSize: requestSearchSettings.gridSize,
                        thetaGridSize: requestSearchSettings.thetaGridSize,
                        residualThreshold: requestSearchSettings.residualThreshold,
                        useContinuation: requestSearchSettings.useContinuation
                    },
                    customEquations: activeAppliedCustomEquations,
                    customParams: appliedParamValidation.normalized
                });
                if (cancelled) return;

                ulamSupportRef.current = dynamicSystem === 'henon' ? (result?.support || null) : null;
                setPeriodicState(prev => ({
                    ...prev,
                    orbits: result?.orbits || [],
                    isReady: true,
                    computeMethod: result?.usedContinuation ? 'continuation' : 'grid',
                    resultRevision
                }));
            } catch (err) {
                if (cancelled) return;
                console.error('Failed to compute periodic orbits:', err);
                ulamSupportRef.current = null;
                setPeriodicState(prev => ({
                    ...prev,
                    isReady: true,
                    orbits: [],
                    computeMethod: null,
                    resultRevision
                }));
            }
        };

        initSystem();
        return () => { cancelled = true; };
    }, [wasmModule, dynamicSystem, params.a, params.b, params.delta, params.h, params.epsilon, params.maxPeriod, params.startX, params.startY, viewRange, periodicSearchSettings.gridSize, periodicSearchSettings.thetaGridSize, periodicSearchSettings.residualThreshold, periodicSearchSettings.useContinuation, activeAppliedCustomEquations, appliedParamValidation, computeRequestId, runComputeTask]);

    useEffect(() => {
        if (manifoldDebounceRef.current) {
            clearTimeout(manifoldDebounceRef.current);
        }
        let cancelled = false;
        const sourcePeriodicRevision = periodicState.resultRevision;

        setManifoldState(prev => ({ ...prev, isComputing: true, rawManifolds: [] }));

        manifoldDebounceRef.current = setTimeout(() => {
            if (!wasmModule) {
                setManifoldState(prev => ({
                    ...prev,
                    isComputing: false,
                    sourcePeriodicRevision
                }));
                return;
            }
            const support = dynamicSystem === 'henon' ? ulamSupportRef.current : null;

            const manifoldsEnabled = manifoldState.showUnstableManifold || manifoldState.showStableManifold;

            if (!manifoldsEnabled && (dynamicSystem === 'henon' || dynamicSystem === 'duffing' || dynamicSystem === 'custom')) {
                const orbits = periodicState.orbits || [];
                const fixedPoints = fixedPointSolutionsFromOrbits(orbits);

                setManifoldState(prev => ({
                    ...prev,
                    manifolds: [],
                    rawManifolds: [],
                    stableManifolds: [],
                    fixedPoints: fixedPoints,
                    intersections: [],
                    isComputing: false,
                    isReady: true,
                    sourcePeriodicRevision
                }));
                return;
            }
            if ((dynamicSystem === 'custom' || dynamicSystem === 'custom_ode') && !appliedParamValidation.valid) {
                setManifoldState(prev => ({
                    ...prev,
                    isComputing: false,
                    isReady: true,
                    manifolds: [],
                    rawManifolds: [],
                    stableManifolds: [],
                    fixedPoints: [],
                    intersections: [],
                    sourcePeriodicRevision
                }));
                return;
            }
            if (dynamicSystem === 'custom_ode') {
                console.log('Initializing user-defined ODE BDE flow simulation');
                if (bdeSimRef.current) {
                    bdeSimRef.current.free();
                }
                bdeSimRef.current = new wasmModule.BdeSimulatorUserDefinedWasm(
                    activeAppliedCustomEquations.xEq, activeAppliedCustomEquations.yEq,
                    appliedParamValidation.normalized,
                    params.epsilon,
                    manifoldState.startPoint.x, manifoldState.startPoint.y, 0.05, 1000
                );
                const simulator = bdeSimRef.current;
                setBdeState(prev => ({
                    ...prev,
                    points: simulator.get_points() as ExtendedState[],
                    isRunning: false,
                }));
                if (bdeAnimRef.current) cancelAnimationFrame(bdeAnimRef.current);

                setManifoldState(prev => ({
                    ...prev,
                    manifolds: [],
                    rawManifolds: [],
                    stableManifolds: [],
                    fixedPoints: [],
                    isComputing: false,
                    isReady: true,
                    sourcePeriodicRevision
                }));
                return;
            }
            if (dynamicSystem === 'duffing_ode') {
                try {
                    console.log('Initializing Duffing ODE BDE flow simulation');
                    if (bdeSimRef.current) {
                        bdeSimRef.current.free();
                    }
                    bdeSimRef.current = new wasmModule.BdeSimulatorWasm(
                        params.delta,
                        params.epsilon,
                        manifoldState.startPoint.x,
                        manifoldState.startPoint.y,
                        CONTINUOUS_BOUNDARY_FLOW_SETTINGS.initialRadius,
                        CONTINUOUS_BOUNDARY_FLOW_SETTINGS.sampleCount
                    );
                    const simulator = bdeSimRef.current;
                    setBdeState(prev => ({
                        ...prev,
                        points: simulator.get_points() as ExtendedState[],
                        isRunning: false,
                    }));
                    if (bdeAnimRef.current) cancelAnimationFrame(bdeAnimRef.current);

                    setManifoldState(prev => ({
                        ...prev,
                        manifolds: [],
                        rawManifolds: [],
                        stableManifolds: [],
                        fixedPoints: [],
                        isComputing: false,
                        isReady: true,
                        sourcePeriodicRevision
                    }));
                } catch (err) {
                    console.error('Manifold computation error:', err);
                    setManifoldState(prev => ({
                        ...prev,
                        isComputing: false,
                        sourcePeriodicRevision
                    }));
                }
                return;
            }

            runComputeTask('computeManifolds', {
                dynamicSystem,
                params: {
                    a: params.a,
                    b: params.b,
                    epsilon: params.epsilon
                },
                viewRange: {
                    xMin: viewRange.xMin,
                    xMax: viewRange.xMax,
                    yMin: viewRange.yMin,
                    yMax: viewRange.yMax
                },
                periodicOrbits: (() => {
                    const allSaddlesAndRepellers = (periodicState.orbits || []).filter(o => {
                        const stab = (o.stability || '').toLowerCase();
                        return stab === 'saddle' || stab === 'unstable' || stab === 'dualrepeller' || stab === 'dual_repeller';
                    });
                    const candidateOrbits = allSaddlesAndRepellers.length > 0 ? allSaddlesAndRepellers : (periodicState.orbits || []);
                    if (manifoldState.selectedOrbitPeriod === undefined || manifoldState.selectedOrbitPeriod === 'all') {
                        return candidateOrbits;
                    }
                    const periodNum = typeof manifoldState.selectedOrbitPeriod === 'string'
                        ? Number(manifoldState.selectedOrbitPeriod)
                        : manifoldState.selectedOrbitPeriod;
                    const periodMatched = candidateOrbits.filter(o => o.period === periodNum);
                    return periodMatched.length > 0 ? periodMatched : candidateOrbits;
                })(),
                customEquations: activeAppliedCustomEquations,
                customParams: appliedParamValidation.normalized,
                showStableManifold: manifoldState.showStableManifold,
                showUnstableManifold: manifoldState.showUnstableManifold,
                intersectionThreshold: manifoldState.intersectionThreshold,
                maximumPointSpacing: manifoldState.maximumManifoldPointSpacing,
            }).then((result) => {
                if (cancelled) return;
                setManifoldState(prev => ({
                    ...prev,
                    manifolds: clipManifoldsBySupport(result?.manifolds || [], support),
                    rawManifolds: result?.manifolds || [],
                    stableManifolds: result?.stableManifolds || [],
                    fixedPoints: enrichSolutionPointsWithOrbitNormals(
                        result?.fixedPoints || [],
                        periodicState.orbits || []
                    ),
                    intersections: result?.intersections || [],
                    isComputing: false,
                    isReady: true,
                    sourcePeriodicRevision
                }));
            }).catch((err) => {
                if (cancelled) return;
                console.error('Manifold computation error:', err);
                setManifoldState(prev => ({
                    ...prev,
                    isComputing: false,
                    sourcePeriodicRevision
                }));
            });
        }, 500);

        return () => {
            cancelled = true;
            if (manifoldDebounceRef.current) {
                clearTimeout(manifoldDebounceRef.current);
            }
        };
    }, [dynamicSystem, params.a, params.b, params.delta, params.h, params.epsilon, periodicState.orbits, periodicState.resultRevision, wasmModule, manifoldState.showStableManifold, manifoldState.showUnstableManifold, manifoldState.intersectionThreshold, manifoldState.maximumManifoldPointSpacing, manifoldState.selectedOrbitPeriod, activeAppliedCustomEquations, appliedParamValidation, manifoldState.startPoint.x, manifoldState.startPoint.y, viewRange, computeRequestId, runComputeTask]);

    useEffect(() => {
        if (!animationState.isAnimating || animationState.awaitingResult) {
            return;
        }

        if (!periodicState.isReady || manifoldState.isComputing) {
            return;
        }

        const nextStep = nextParameterAnimationStep(animationState);
        if (!nextStep) {
            animationSearchSettingsRef.current = null;
            setAnimationState(prev => ({
                ...prev,
                isAnimating: false,
                isPreparing: false,
                expectedPeriodicRevision: null
            }));
            return;
        }

        const expectedPeriodicRevision = periodicComputationRevisionRef.current + 1;

        setAnimationState(prev => ({
            ...prev,
            currentStep: nextStep.step,
            awaitingResult: true,
            expectedPeriodicRevision
        }));
        setParams(previous => applyParameterAnimationValue(
            previous,
            animationState.parameter,
            nextStep.value
        ));
        setDraftParams(previous => applyParameterAnimationValue(
            previous,
            animationState.parameter,
            nextStep.value
        ));
    }, [animationState, manifoldState.isComputing, periodicState.isReady]);

    const beginAnimationFromAppliedState = useCallback(async () => {
        if (
            !periodicState.isReady
            || manifoldState.isComputing
            || periodicState.renderedRevision !== periodicState.resultRevision
            || manifoldState.sourcePeriodicRevision !== periodicState.resultRevision
        ) return;

        const baseVal = params[animationState.parameter];
        const targetVal = baseVal + (animationState.direction * animationState.rangeValue);
        animationSearchSettingsRef.current = capturePeriodicSearchSettings(
            periodicSearchSettings
        );

        if (recordingState.recordingEnabled && canvasRef.current) {
            recordedFramesRef.current = [];
            setRecordingState(prev => ({
                ...prev,
                isRecording: true,
                frameCount: 0,
                error: null
            }));
            try {
                const canvas = canvasRef.current;
                const width = 1280;
                const height = 720;
                const offscreen = new OffscreenCanvas(width, height);
                const ctx = offscreen.getContext('2d');
                if (!ctx) throw new Error('Could not create the recording canvas context.');
                ctx.drawImage(canvas, 0, 0, width, height);

                ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                ctx.fillRect(10, height - 40, 500, 30);
                ctx.font = 'bold 16px monospace';
                ctx.fillStyle = '#4CAF50';
                ctx.fillText(`${systemLabel} | ${paramOverlayText}`, 20, height - 18);

                const bitmap = await createImageBitmap(offscreen);
                recordedFramesRef.current.push(bitmap);
                setRecordingState(prev => ({ ...prev, frameCount: 1 }));
                console.log('Initial frame captured');
            } catch (err) {
                console.error('Initial frame capture error:', err);
            }
        }

        setAnimationState(prev => ({
            ...prev,
            isAnimating: true,
            isPreparing: false,
            baseValue: baseVal,
            targetValue: targetVal,
            currentStep: 0,
            awaitingResult: false,
            expectedPeriodicRevision: null
        }));
    }, [params, animationState.parameter, animationState.direction, animationState.rangeValue, recordingState.recordingEnabled, systemLabel, paramOverlayText, periodicState.isReady, periodicState.resultRevision, periodicState.renderedRevision, periodicSearchSettings, manifoldState.isComputing, manifoldState.sourcePeriodicRevision]);

    const startAnimation = useCallback(() => {
        if (animationState.isPreparing) return;
        if (hasPendingInputChanges) {
            pendingAnimationStartRef.current = true;
            setAnimationState(previous => ({
                ...previous,
                isPreparing: true
            }));
            const applied = applyInputsAndRecompute();
            if (!applied) {
                pendingAnimationStartRef.current = false;
                setAnimationState(previous => ({
                    ...previous,
                    isPreparing: false
                }));
            }
            return;
        }
        return beginAnimationFromAppliedState();
    }, [
        animationState.isPreparing,
        applyInputsAndRecompute,
        beginAnimationFromAppliedState,
        hasPendingInputChanges
    ]);

    useEffect(() => {
        if (!pendingAnimationStartRef.current) return;
        if (hasPendingInputChanges) return;
        if (
            !periodicState.isReady
            || manifoldState.isComputing
            || periodicState.renderedRevision !== periodicState.resultRevision
            || manifoldState.sourcePeriodicRevision !== periodicState.resultRevision
        ) return;

        pendingAnimationStartRef.current = false;
        void beginAnimationFromAppliedState();
    }, [
        beginAnimationFromAppliedState,
        hasPendingInputChanges,
        manifoldState.isComputing,
        manifoldState.sourcePeriodicRevision,
        periodicState.isReady,
        periodicState.renderedRevision,
        periodicState.resultRevision
    ]);

    const stopAnimation = useCallback(() => {
        pendingAnimationStartRef.current = false;
        animationSearchSettingsRef.current = null;
        setAnimationState(prev => ({
            ...prev,
            isAnimating: false,
            isPreparing: false,
            currentStep: 0,
            awaitingResult: false,
            expectedPeriodicRevision: null
        }));
    }, []);

    const captureFrame = useCallback(async () => {
        if (!canvasRef.current || !recordingState.recordingEnabled) return null;

        const canvas = canvasRef.current;
        const width = 1280;
        const height = 720;

        const offscreen = new OffscreenCanvas(width, height);
        const ctx = offscreen.getContext('2d');
        if (!ctx) throw new Error('Could not create the frame-capture canvas context.');

        ctx.drawImage(canvas, 0, 0, width, height);

        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(10, height - 40, 500, 30);
        ctx.font = 'bold 16px monospace';
        ctx.fillStyle = '#4CAF50';
        ctx.fillText(`${systemLabel} | ${paramOverlayText}`, 20, height - 18);

        const bitmap = await createImageBitmap(offscreen);
        return bitmap;
    }, [recordingState.recordingEnabled, systemLabel, paramOverlayText]);

    const generateFilename = useCallback(() => {
        const paramName = animationState.parameter;
        const startStr = (animationState.baseValue || 0).toFixed(3).replace('.', 'p').replace('-', 'm');
        const endStr = (animationState.targetValue || 0).toFixed(3).replace('.', 'p').replace('-', 'm');

        return `${systemFilePrefix}_${paramName}_${paramFileString}_${startStr}_to_${endStr}.mp4`;
    }, [systemFilePrefix, paramFileString, animationState.parameter, animationState.baseValue, animationState.targetValue]);

    const initEncoderWorker = useCallback(() => {
        if (encoderWorkerRef.current) {
            encoderWorkerRef.current.terminate();
        }

        const worker = new Worker(
            new URL('./videoEncoder.worker.ts', import.meta.url),
            { type: 'module' }
        );

        worker.onmessage = (event: MessageEvent<UnknownRecord>) => {
            const { type, blob, frameCount, error } = event.data;

            switch (type) {
                case 'ready':
                    console.log('Encoder ready');
                    break;
                case 'progress':
                    setRecordingState(prev => ({
                        ...prev,
                        frameCount: typeof frameCount === 'number' ? frameCount : prev.frameCount,
                    }));
                    break;
                case 'complete': {
                    if (!(blob instanceof Blob)) break;
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    const filename = generateFilename();
                    a.href = url;
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);

                    setRecordingState(prev => ({
                        ...prev,
                        isEncoding: false,
                        isRecording: false,
                        recordingEnabled: false
                    }));
                    recordedFramesRef.current = [];
                    break;
                }
                case 'error':
                    console.error('Encoder error:', error);
                    setRecordingState(prev => ({
                        ...prev,
                        error: typeof error === 'string' ? error : 'Video encoder failed.',
                        isEncoding: false,
                    }));
                    break;
            }
        };

        encoderWorkerRef.current = worker;
        return worker;
    }, [generateFilename]);

    const savePNG = useCallback(async () => {
        if (!canvasRef.current || !rendererRef.current || !sceneRef.current || !cameraRef.current) return;

        // Force a render to ensure canvas has the current frame
        rendererRef.current.render(sceneRef.current, cameraRef.current);

        const canvas = canvasRef.current;
        const width = 1920;
        const height = 1080;

        const offscreen = new OffscreenCanvas(width, height);
        const ctx = offscreen.getContext('2d');
        if (!ctx) throw new Error('Could not create the PNG export canvas context.');

        ctx.drawImage(canvas, 0, 0, width, height);

        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(10, height - 80, 500, 70);

        ctx.font = 'bold 18px monospace';
        ctx.fillStyle = '#4CAF50';

        ctx.fillText(`${systemLabel} | Iteration: ${manifoldState.iteration}`, 20, height - 55);
        ctx.font = '14px monospace';
        ctx.fillStyle = '#aaa';
        const unstablePts = manifoldState.manifolds.reduce((sum, manifold) => (
            sum
            + (manifold.plus?.points?.length || 0)
            + (manifold.minus?.points?.length || 0)
        ), 0);
        const orbitsInfo = periodicState.orbits.length > 0 ? `${periodicState.orbits.length} orbits, ` : '';
        ctx.fillText(`${orbitsInfo}${manifoldState.fixedPoints.length} fixed pts, ${unstablePts} manifold pts`, 20, height - 32);

        ctx.font = 'bold 14px monospace';
        ctx.fillStyle = '#4CAF50';
        ctx.fillText(paramOverlayText, 20, height - 12);

        const blob = await offscreen.convertToBlob({ type: 'image/png', quality: 1.0 });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');

        const iterStr = manifoldState.hasStarted ? `_iter${manifoldState.iteration}` : '';

        a.href = url;
        a.download = `${systemFilePrefix}_${paramFileString}${iterStr}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, [periodicState.orbits.length, manifoldState.iteration, manifoldState.hasStarted, manifoldState.manifolds, manifoldState.fixedPoints, systemLabel, paramOverlayText, systemFilePrefix, paramFileString]);

    const startEncoding = useCallback(async () => {
        if (recordedFramesRef.current.length === 0) {
            console.warn('No frames to encode');
            return;
        }

        setRecordingState(prev => ({ ...prev, isEncoding: true, encodingProgress: 0 }));

        const worker = initEncoderWorker();

        worker.postMessage({
            type: 'init',
            data: {
                width: 1280,
                height: 720,
                fps: 2
            }
        });

        await new Promise(resolve => setTimeout(resolve, 100));

        const fps = 2;
        const frameDuration = 1000000 / fps; // microseconds (500ms per frame)

        for (let i = 0; i < recordedFramesRef.current.length; i++) {
            const frame = recordedFramesRef.current[i];
            worker.postMessage({
                type: 'frame',
                data: {
                    imageData: frame,
                    timestamp: i * frameDuration,
                    duration: frameDuration
                }
            });

            if (i % 5 === 0) {
                await new Promise(resolve => setTimeout(resolve, 10));
            }
        }

        worker.postMessage({ type: 'finish' });
    }, [initEncoderWorker]);

    useEffect(() => {
        if (!isParameterAnimationStepSettled({
            animationState,
            periodicState,
            manifoldState
        })) {
            return;
        }

        let cancelled = false;
        const expectedRevision = animationState.expectedPeriodicRevision;
        const frameRequest = requestAnimationFrame(async () => {
            try {
                if (recordingState.recordingEnabled) {
                    const frame = await captureFrame();
                    if (frame && !cancelled) {
                        recordedFramesRef.current.push(frame);
                        setRecordingState(prev => ({ ...prev, frameCount: recordedFramesRef.current.length }));
                    }
                }
            } catch (err) {
                console.error('[Recording] Frame capture error:', err);
                if (!cancelled) {
                    setRecordingState(prev => ({
                        ...prev,
                        error: err instanceof Error ? err.message : String(err)
                    }));
                }
            } finally {
                if (!cancelled) {
                    setAnimationState(prev => (
                        prev.expectedPeriodicRevision === expectedRevision
                            ? {
                                ...prev,
                                awaitingResult: false,
                                expectedPeriodicRevision: null
                            }
                            : prev
                    ));
                }
            }
        });

        return () => {
            cancelled = true;
            cancelAnimationFrame(frameRequest);
        };
    }, [animationState, periodicState, manifoldState, recordingState.recordingEnabled, captureFrame]);

    useEffect(() => {
        if (!animationState.isAnimating && recordingState.recordingEnabled && recordedFramesRef.current.length > 0 && !recordingState.isEncoding) {
            console.log(`[Recording] Animation finished with ${recordedFramesRef.current.length} frames, starting encoding...`);
            startEncoding();
        }
    }, [animationState.isAnimating, recordingState.recordingEnabled, recordingState.isEncoding, startEncoding]);

    // Toggle recording mode
    const toggleRecording = useCallback(() => {
        if (recordingState.recordingEnabled) {
            // Disable recording
            setRecordingState(prev => ({ ...prev, recordingEnabled: false }));
            recordedFramesRef.current = [];
        } else {
            // Enable recording
            setRecordingState(prev => ({ ...prev, recordingEnabled: true, frameCount: 0, error: null }));
            recordedFramesRef.current = [];
        }
    }, [recordingState.recordingEnabled]);

    const computeForwardInvariantSet = useCallback(async () => {
        if (dynamicSystem !== 'henon') return;
        let seed: ExtendedState;
        try {
            if (invariantSetState.seedMode === 'manual') {
                seed = { ...manifoldState.startPoint };
            } else {
                const position = randomHenonInvariantSeed(
                    viewRange,
                    params.a,
                    params.b,
                    params.epsilon,
                );
                seed = {
                    ...position,
                    nx: manifoldState.startPoint.nx,
                    ny: manifoldState.startPoint.ny,
                };
            }
        } catch (error) {
            setInvariantSetState(previous => ({
                ...previous,
                error: error instanceof Error ? error.message : String(error),
            }));
            return;
        }

        const requestId = ++invariantSetRequestIdRef.current;
        setInvariantSetState(previous => ({
            ...previous,
            isComputing: true,
            result: null,
            error: null,
        }));
        try {
            const result = await runComputeTask('computeForwardInvariantSet', {
                seed,
                params: { a: params.a, b: params.b, epsilon: params.epsilon },
                domain: viewRange,
                settings: {
                    boundaryPointCount: invariantSetState.boundaryPointCount,
                    forwardIterations: invariantSetState.forwardIterations,
                },
            });
            if (requestId !== invariantSetRequestIdRef.current) return;
            setInvariantSetState(previous => ({
                ...previous,
                isComputing: false,
                showResult: true,
                result,
                error: null,
            }));
        } catch (error) {
            if (requestId !== invariantSetRequestIdRef.current) return;
            setInvariantSetState(previous => ({
                ...previous,
                isComputing: false,
                result: null,
                error: error instanceof Error ? error.message : String(error),
            }));
        }
    }, [
        dynamicSystem,
        invariantSetState.boundaryPointCount,
        invariantSetState.forwardIterations,
        invariantSetState.seedMode,
        manifoldState.startPoint.x,
        manifoldState.startPoint.y,
        manifoldState.startPoint.nx,
        manifoldState.startPoint.ny,
        params.a,
        params.b,
        params.epsilon,
        runComputeTask,
        viewRange,
    ]);

    const computeGeometricOffsets = useCallback(async () => {
        if (dynamicSystem !== 'henon' || calculatedBoundaryBranches.length === 0) {
            setGeometricOffsetState(prev => ({
                ...prev,
                error: 'At least one unstable-manifold boundary point with an attached normal is required.'
            }));
            return;
        }
        const requestId = ++geometricOffsetRequestIdRef.current;
        inverseOffsetRequestIdRef.current += 1;
        setGeometricOffsetState(prev => ({
            ...prev,
            isComputing: true,
            contours: prev.contours.map(contour => ({
                ...contour,
                result: null,
                inverseResult: null,
                error: null,
                inverseError: null,
            })),
            error: null,
            inverseError: null
        }));
        try {
            const request = buildGeometricOffsetBatchRequest(
                calculatedBoundaryBranches.map(points => ({
                    points,
                    isClosed: hasVerifiedBoundaryCycles,
                })),
                geometricOffsetState.contours,
            );
            const batch = await runComputeTask('computeGeometricOffsetBatch', request);
            if (requestId !== geometricOffsetRequestIdRef.current) return;
            if (batch.contours.length !== geometricOffsetState.contours.length
                || batch.contours.some(({ result }) => (
                    result?.completed_levels !== 1 || result?.levels?.length !== 1
                ))) {
                throw new Error('The direct geometric contour batch was not completed.');
            }
            const resultsById = new Map(batch.contours.map(item => [item.id, item.result]));
            setGeometricOffsetState(prev => ({
                ...prev,
                isComputing: false,
                contours: prev.contours.map(contour => ({
                    ...contour,
                    result: resultsById.get(contour.id) ?? null,
                    inverseResult: null,
                    error: resultsById.has(contour.id) ? null : 'No result was returned for this contour.',
                    inverseError: null,
                })),
                error: null,
                inverseError: null,
            }));
            const contourCurves = batch.contours.flatMap(({ result }) => (
                result.levels.flatMap(level => (
                    (level.boundary_components || []).map((component, componentIndex) => ({
                        inverse_iteration: 0,
                        source_component_id: component.id ?? componentIndex,
                        points: component.points.flatMap(point => {
                            if (!Array.isArray(point)) return [point];
                            const [x, y, nx = 0, ny = 0] = point;
                            return [x, y].every(Number.isFinite)
                                ? [{ x, y, nx, ny }]
                                : [];
                        }),
                    }))
                ))
            ));
            const viewportSize = readViewportSize();
            const fittedRange = fitInverseOffsetCurveRange(
                contourCurves,
                viewportSize.width / viewportSize.height,
                viewRange,
            );
            if (fittedRange) transitionViewRange(fittedRange);
        } catch (error) {
            if (requestId !== geometricOffsetRequestIdRef.current) return;
            setGeometricOffsetState(prev => ({
                ...prev,
                isComputing: false,
                contours: prev.contours.map(contour => ({
                    ...contour,
                    result: null,
                    inverseResult: null,
                    error: error instanceof Error ? error.message : String(error),
                    inverseError: null,
                })),
                error: error instanceof Error ? error.message : String(error),
                inverseError: null
            }));
        }
    }, [
        dynamicSystem,
        calculatedBoundaryBranches,
        geometricOffsetState.contours,
        hasVerifiedBoundaryCycles,
        readViewportSize,
        runComputeTask,
        viewRange,
        transitionViewRange
    ]);

    const computeInverseGeometricOffsets = useCallback(async () => {
        const sources = geometricOffsetSourceContours(geometricOffsetState);
        if (dynamicSystem !== 'henon' || sources.length === 0) {
            setGeometricOffsetState(previous => ({
                ...previous,
                inverseError: 'Compute at least one geometric offset source first.'
            }));
            return;
        }
        let batchSources;
        try {
            batchSources = sources.map(contour => ({
                id: contour.id,
                levels: contour.result!.levels.slice(0, 1),
                positionTolerance: inverseOffsetPositionTolerance(
                    geometricOffsetSampleSpacing(contour.result),
                ),
            }));
        } catch (error) {
            setGeometricOffsetState(previous => ({
                ...previous,
                inverseError: error instanceof Error ? error.message : String(error)
            }));
            return;
        }
        const requestId = ++inverseOffsetRequestIdRef.current;
        setGeometricOffsetState(previous => ({
            ...previous,
            isComputingInverse: true,
            contours: previous.contours.map(contour => sources.some(source => source.id === contour.id)
                ? { ...contour, inverseResult: null, inverseError: null }
                : contour),
            inverseError: null
        }));
        try {
            const batch = await runComputeTask('computeInverseGeometricOffsetBatch', {
                sources: batchSources,
                params: { a: params.a, b: params.b, epsilon: params.epsilon },
                domain: viewRange,
                settings: {
                    iterations: geometricOffsetState.inverseIterations,
                    normalTolerance: DEFAULT_GEOMETRIC_OFFSET_SETTINGS.inverseNormalTolerance,
                    maxSubdivisionDepth: DEFAULT_GEOMETRIC_OFFSET_SETTINGS.inverseMaximumSubdivisionDepth
                }
            });
            if (requestId !== inverseOffsetRequestIdRef.current) return;
            if (batch.sources.length !== sources.length) {
                throw new Error('The inverse geometric contour batch was not completed.');
            }
            const resultsById = new Map(batch.sources.map(source => [source.id, source.result]));
            setGeometricOffsetState(previous => ({
                ...previous,
                isComputingInverse: false,
                contours: previous.contours.map(contour => resultsById.has(contour.id)
                    ? {
                        ...contour,
                        inverseResult: resultsById.get(contour.id) ?? null,
                        inverseError: null,
                    }
                    : contour),
                inverseError: null,
                showInverseContours: true
            }));
            const visibleCurves = batch.sources.flatMap(source => visibleInverseOffsetCurves(
                source.result,
                geometricOffsetState.inverseDisplayMode,
            ));
            const viewportSize = readViewportSize();
            const fittedRange = fitInverseOffsetCurveRange(
                visibleCurves,
                viewportSize.width / viewportSize.height,
                viewRange,
            );
            if (fittedRange) transitionViewRange(fittedRange);
        } catch (error) {
            if (requestId !== inverseOffsetRequestIdRef.current) return;
            setGeometricOffsetState(previous => ({
                ...previous,
                isComputingInverse: false,
                contours: previous.contours.map(contour => sources.some(source => source.id === contour.id)
                    ? {
                        ...contour,
                        inverseResult: null,
                        inverseError: error instanceof Error ? error.message : String(error),
                    }
                    : contour),
                inverseError: error instanceof Error ? error.message : String(error)
            }));
        }
    }, [
        dynamicSystem,
        geometricOffsetState,
        params.a,
        params.b,
        params.epsilon,
        viewRange,
        readViewportSize,
        runComputeTask,
        transitionViewRange
    ]);

    const fitInverseGeometricOffsets = useCallback(() => {
        const sourceIds = new Set(geometricOffsetState.preimageSourceIds);
        if (geometricOffsetState.selectedContourId) {
            sourceIds.add(geometricOffsetState.selectedContourId);
        }
        const curves = geometricOffsetState.contours
            .filter(contour => sourceIds.has(contour.id))
            .flatMap(contour => visibleInverseOffsetCurves(
                contour.inverseResult,
                geometricOffsetState.inverseDisplayMode,
            ));
        const { width: viewportWidth, height: viewportHeight } = readViewportSize();
        const fittedRange = fitInverseOffsetCurveRange(
            curves,
            viewportWidth / viewportHeight,
            viewRange,
        );
        if (fittedRange) transitionViewRange(fittedRange);
    }, [
        geometricOffsetState.contours,
        geometricOffsetState.inverseDisplayMode,
        geometricOffsetState.preimageSourceIds,
        geometricOffsetState.selectedContourId,
        readViewportSize,
        viewRange,
        transitionViewRange
    ]);

    useEffect(() => {
        invariantSetRequestIdRef.current += 1;
        setInvariantSetState(previous => ({
            ...previous,
            isComputing: false,
            result: null,
            error: null,
        }));
    }, [dynamicSystem, params.a, params.b, params.epsilon, viewRange]);

    useEffect(() => {
        if (invariantSetState.seedMode !== 'manual') return;
        invariantSetRequestIdRef.current += 1;
        setInvariantSetState(previous => ({
            ...previous,
            isComputing: false,
            result: null,
            error: null,
        }));
    }, [
        invariantSetState.seedMode,
        manifoldState.startPoint.x,
        manifoldState.startPoint.y,
    ]);

    useEffect(() => {
        geometricOffsetRequestIdRef.current += 1;
        inverseOffsetRequestIdRef.current += 1;
        setGeometricOffsetState(prev => ({
            ...prev,
            isComputing: false,
            isComputingInverse: false,
            contours: prev.contours.map(contour => ({
                ...contour,
                result: null,
                inverseResult: null,
                error: null,
                inverseError: null,
            })),
            error: null,
            inverseError: null
        }));
    }, [params.a, params.b, params.epsilon, manifoldState.manifolds]);

    useEffect(() => {
        if (!sceneRef.current) return;
        const scene = sceneRef.current;

        const toRemove: THREE.Object3D[] = [];
        scene.traverse(child => {
            if (child.userData.type === 'trajectory' || child.userData.type === 'manifold' || child.userData.type === 'boundaryLayer' || child.userData.type === 'boundarySamplePoints' || child.userData.type === 'noiseBall' || child.userData.type === 'geometricOffset' || child.userData.type === 'inverseGeometricOffset' || child.userData.type === 'forwardInvariantSet' || child.userData.type === 'fixedPoint' || child.userData.type === 'bde') {
                toRemove.push(child);
            }
        });
        toRemove.forEach(obj => {
            if (obj instanceof THREE.Mesh || obj instanceof THREE.Line || obj instanceof THREE.Points || obj instanceof Line2) {
                obj.geometry.dispose();
                if (Array.isArray(obj.material)) obj.material.forEach(material => material.dispose());
                else obj.material.dispose();
            }
            scene.remove(obj);
        });

        const addBoundaryLine = (
            points: ProjectedState[],
            color: string,
            linewidth: number,
            zPosition: number,
            boundaryRole: string,
            closeCycle = false,
        ): void => {
            if (points.length < 2) return;
            const renderedPoints = closeCycle ? [...points, points[0]] : points;
            const geometry = new LineGeometry();
            geometry.setPositions(renderedPoints.flatMap(point => [point.x, point.y, zPosition]));
            const material = new LineMaterial({
                color,
                linewidth,
                transparent: true,
                opacity: 0.98,
                depthTest: false,
            });
            const viewportSize = readViewportSize();
            material.resolution.set(viewportSize.width, viewportSize.height);
            const line = new Line2(geometry, material);
            line.computeLineDistances();
            line.renderOrder = 10;
            line.userData = { type: 'boundaryLayer', boundaryRole };
            scene.add(line);
        };

        const addBoundarySamplePoints = (
            points: ProjectedState[],
            color: string,
            zPosition: number,
            boundaryRole: string,
        ): void => {
            if (points.length < 1) return;
            const geometry = new THREE.BufferGeometry().setFromPoints(
                points.map(point => new THREE.Vector3(point.x, point.y, zPosition)),
            );
            const material = new THREE.PointsMaterial({
                color: new THREE.Color(color),
                size: 4,
                sizeAttenuation: false,
                transparent: true,
                opacity: 0.9,
                depthTest: false,
                depthWrite: false,
            });
            const samples = new THREE.Points(geometry, material);
            samples.renderOrder = 12;
            samples.userData = { type: 'boundarySamplePoints', boundaryRole };
            scene.add(samples);
        };

        if (invariantSetState.showResult && invariantSetState.result?.point_sets.length) {
            // Iteration zero is the construction circle, not a forward-projected result.
            const projectedPointSets = invariantSetState.result.point_sets.filter(pointSet => (
                pointSet.iteration > 0
            ));
            const projectedPoints = projectedPointSets.flatMap(pointSet => pointSet.points);
            if (projectedPoints.length > 0) {
                const geometry = new THREE.BufferGeometry().setFromPoints(
                    projectedPoints.map(point => new THREE.Vector3(point.x, point.y, 0.31)),
                );
                const material = new THREE.PointsMaterial({
                    color: new THREE.Color('#70a7c4'),
                    size: 4,
                    sizeAttenuation: false,
                    transparent: true,
                    opacity: 0.9,
                    depthTest: false,
                    depthWrite: false,
                });
                const samples = new THREE.Points(geometry, material);
                samples.renderOrder = 15;
                samples.userData = {
                    type: 'forwardInvariantSet',
                    role: 'projected-point-samples',
                    pointSetCount: projectedPointSets.length,
                    pointCount: projectedPoints.length,
                };
                scene.add(samples);
            }
        }

        if (manifoldState.showUnstableManifold && hasBoundarySamples) {
            boundaryLayers.invariantBranches.forEach(branch => {
                if (manifoldState.showBoundarySamplePoints) {
                    addBoundarySamplePoints(
                        branch,
                        BOUNDARY_LAYER_COLORS.invariant,
                        0.26,
                        'unstable-manifold',
                    );
                } else {
                    addBoundaryLine(
                        branch,
                        BOUNDARY_LAYER_COLORS.invariant,
                        3.4,
                        0.24,
                        'unstable-manifold',
                        hasVerifiedBoundaryCycles,
                    );
                }
            });

            if (manifoldState.showDeterministicImageBoundary) {
                boundaryLayers.deterministicBranches.forEach(branch => {
                    if (manifoldState.showBoundarySamplePoints) {
                        addBoundarySamplePoints(
                            branch,
                            BOUNDARY_LAYER_COLORS.deterministicImage,
                            0.255,
                            'deterministic-image',
                        );
                    } else {
                        addBoundaryLine(
                            branch,
                            BOUNDARY_LAYER_COLORS.deterministicImage,
                            2.6,
                            0.23,
                            'deterministic-image',
                            hasVerifiedBoundaryCycles,
                        );
                    }
                });
            }

            if (manifoldState.showNoiseBalls && params.epsilon > 0) {
                const centers = boundaryLayers.noiseBallCenters;
                const fillGeometry = new THREE.CircleGeometry(params.epsilon, 32);
                const fillMaterial = new THREE.MeshBasicMaterial({
                    color: new THREE.Color(BOUNDARY_LAYER_COLORS.noiseBall),
                    transparent: true,
                    opacity: 0.003,
                    depthTest: false,
                    depthWrite: false,
                    side: THREE.DoubleSide,
                });
                const fills = new THREE.InstancedMesh(fillGeometry, fillMaterial, centers.length);
                const transform = new THREE.Matrix4();
                centers.forEach((center, index) => {
                    transform.makeTranslation(center.x, center.y, 0.205);
                    fills.setMatrixAt(index, transform);
                });
                fills.instanceMatrix.needsUpdate = true;
                fills.renderOrder = 6;
                fills.userData = { type: 'noiseBall', role: 'fill', count: centers.length };
                scene.add(fills);

                const outlinePoints: THREE.Vector3[] = [];
                centers.forEach(center => {
                    for (let index = 0; index < 32; index += 1) {
                        const start = index * 2 * Math.PI / 32;
                        const end = (index + 1) * 2 * Math.PI / 32;
                        outlinePoints.push(
                            new THREE.Vector3(
                                center.x + params.epsilon * Math.cos(start),
                                center.y + params.epsilon * Math.sin(start),
                                0.215,
                            ),
                            new THREE.Vector3(
                                center.x + params.epsilon * Math.cos(end),
                                center.y + params.epsilon * Math.sin(end),
                                0.215,
                            ),
                        );
                    }
                });
                const outlineGeometry = new THREE.BufferGeometry().setFromPoints(outlinePoints);
                const outlineMaterial = new THREE.LineBasicMaterial({
                    color: new THREE.Color(BOUNDARY_LAYER_COLORS.noiseBall),
                    transparent: true,
                    opacity: 0.035,
                    depthTest: false,
                });
                const outlines = new THREE.LineSegments(outlineGeometry, outlineMaterial);
                outlines.renderOrder = 8;
                outlines.userData = { type: 'noiseBall', role: 'boundary', count: centers.length };
                scene.add(outlines);
            }
        } else if (manifoldState.showUnstableManifold && manifoldState.manifolds.length > 0) {
            manifoldState.manifolds.forEach(manifold => {
                [manifold.plus, manifold.minus].forEach(trajectory => {
                    if (!trajectory?.points || trajectory.points.length <= 1) return;
                    const points = trajectory.points
                        .map(([x, y]) => ({ x, y }))
                        .filter(point => pointInDomain(point, viewRange));
                    if (points.length <= 1) return;
                    if (manifoldState.showBoundarySamplePoints) {
                        addBoundarySamplePoints(
                            points,
                            BOUNDARY_LAYER_COLORS.invariant,
                            0.26,
                            'unstable-manifold',
                        );
                    } else {
                        addBoundaryLine(
                            points,
                            BOUNDARY_LAYER_COLORS.invariant,
                            3.4,
                            0.24,
                            'unstable-manifold',
                        );
                    }
                });
            });
        }

        if (manifoldState.showStableManifold && manifoldState.stableManifolds.length > 0) {
            manifoldState.stableManifolds.forEach(manifold => {
                [manifold.plus, manifold.minus].forEach(trajectory => {
                    if (!trajectory?.points || trajectory.points.length <= 1) return;
                    const points = trajectory.points
                        .map(([x, y]) => ({ x, y }))
                        .filter(point => pointInDomain(point, viewRange));
                    if (points.length <= 1) return;
                    if (manifoldState.showBoundarySamplePoints) {
                        addBoundarySamplePoints(
                            points,
                            ORBIT_COLORS.stableManifold,
                            0.25,
                            'stable-manifold',
                        );
                    } else {
                        addBoundaryLine(
                            points,
                            ORBIT_COLORS.stableManifold,
                            2.6,
                            0.23,
                            'stable-manifold',
                        );
                    }
                });
            });
        }

        geometricOffsetState.contours.forEach((contour, contourIndex) => {
            if (!contour.visible || !contour.result?.levels) return;
            const selected = contour.id === geometricOffsetState.selectedContourId;
            contour.result.levels.forEach((level, levelIndex) => {
                (level.boundary_components || []).forEach(component => {
                    const points = (component.points || []).flatMap(point => {
                        const projected = projectedCoordinates(point);
                        return projected
                            && pointInDomain(projected, viewRange)
                            ? [{ x: projected.x, y: projected.y }]
                            : [];
                    });
                    if (points.length === 0) return;
                    const geometry = new THREE.BufferGeometry();
                    geometry.setAttribute('position', new THREE.Float32BufferAttribute(points.flatMap(point => [
                        point.x,
                        point.y,
                        0.22 + contourIndex * 0.004 + levelIndex * 0.001,
                    ]), 3));

                    const colors = computeContourVertexColors(points);
                    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                    const material = new THREE.PointsMaterial({
                        vertexColors: true,
                        size: selected ? 5.5 : 4,
                        sizeAttenuation: false,
                        transparent: true,
                        opacity: selected ? 1 : 0.9,
                        depthTest: false,
                    });
                    const pointCloud = new THREE.Points(geometry, material);
                    pointCloud.userData = {
                        type: 'geometricOffset',
                        contourId: contour.id,
                        epsilon: contour.epsilon,
                        level: level.level,
                        targetDistance: level.target_distance,
                        selected,
                    };
                    scene.add(pointCloud);
                });
            });
        });

        if (geometricOffsetState.showInverseContours) {
            const sourceIds = new Set(geometricOffsetState.preimageSourceIds);
            if (geometricOffsetState.selectedContourId) {
                sourceIds.add(geometricOffsetState.selectedContourId);
            }
            geometricOffsetState.contours.forEach((contour, contourIndex) => {
                if (!sourceIds.has(contour.id) || !contour.inverseResult?.curves) return;
                const visibleCurves = visibleInverseOffsetCurves(
                    contour.inverseResult,
                    geometricOffsetState.inverseDisplayMode,
                );
                visibleCurves.forEach(curve => {
                    const points = (curve.points || []).filter(point => pointInDomain(point, viewRange));
                    if (points.length === 0) return;
                    const colorResult = computeInverseCurveVertexColors(curve);

                    const geometry = new THREE.BufferGeometry();
                    geometry.setAttribute('position', new THREE.Float32BufferAttribute(
                        points.flatMap(point => [
                            point.x,
                            point.y,
                            0.25 + 0.006 * curve.inverse_iteration + contourIndex * 0.0005,
                        ]),
                        3,
                    ));
                    geometry.setAttribute('color', new THREE.BufferAttribute(colorResult.colors, 3));

                    const pointCloud = new THREE.Points(geometry, new THREE.PointsMaterial({
                        vertexColors: true,
                        size: contour.id === geometricOffsetState.selectedContourId ? 5.5 : 4,
                        sizeAttenuation: false,
                        transparent: true,
                        opacity: Math.max(0.75, 0.98 - 0.035 * (curve.inverse_iteration - 1)),
                        depthTest: false,
                    }));
                    pointCloud.userData = {
                        type: 'inverseGeometricOffset',
                        contourId: contour.id,
                        inverseIteration: curve.inverse_iteration,
                        sourceLevel: curve.source_level,
                        sourceComponentId: curve.source_component_id,
                    };
                    scene.add(pointCloud);

                    if (curve.is_closed && points.length >= 3 && geometricOffsetState.inverseColorMode === 'uniform') {
                        const closedPoints = [...points, points[0]];
                        const lineGeometry = new LineGeometry();
                        lineGeometry.setPositions(closedPoints.flatMap(point => [
                            point.x,
                            point.y,
                            0.25 + 0.006 * curve.inverse_iteration + contourIndex * 0.0005,
                        ]));
                        const lineMaterial = new LineMaterial({
                            color: inverseOffsetCurveColor(
                                contourIndex,
                                geometricOffsetState.contours.length,
                                curve.inverse_iteration,
                            ),
                            linewidth: contour.id === geometricOffsetState.selectedContourId ? 3.2 : 2.5,
                            transparent: true,
                            opacity: Math.max(0.72, 0.98 - 0.035 * (curve.inverse_iteration - 1)),
                            depthTest: false,
                        });
                        const viewportSize = readViewportSize();
                        lineMaterial.resolution.set(viewportSize.width, viewportSize.height);
                        const line = new Line2(lineGeometry, lineMaterial);
                        line.computeLineDistances();
                        line.userData = {
                            type: 'inverseGeometricOffset',
                            contourId: contour.id,
                            epsilon: contour.epsilon,
                            sourceLevel: curve.source_level,
                            sourceComponentId: curve.source_component_id,
                            inverseIteration: curve.inverse_iteration,
                        };
                        scene.add(line);
                    }
                });
            });
        }

        if (dynamicSystem === 'duffing_ode' && bdeState.points && bdeState.points.length > 1) {
            const pts = bdeState.points.map(p => new THREE.Vector3(p.x, p.y, 0.15));
            pts.push(pts[0].clone());
            const geom = new THREE.BufferGeometry().setFromPoints(pts);
            const mat = new THREE.LineBasicMaterial({
                color: new THREE.Color('#3d5afe'),
                linewidth: 2,
                transparent: true,
                opacity: 0.85
            });
            const line = new THREE.Line(geom, mat);
            line.userData.type = 'bde';
            scene.add(line);
        }

        const ySpan = Math.abs(viewportRange.yMax - viewportRange.yMin);
        const { height: vpHeight } = readViewportSize();
        const frustumH = Math.max(1e-6, ySpan + 0.24);
        const pxPerUnit = vpHeight / frustumH;
        const radius = Math.max(1e-6, 6.0 / pxPerUnit);

        if (manifoldState.showOrbits) {
            const isFpVisible = (fp: { period?: number; x: number; y: number }): boolean => {
                const p = fp.period ?? 1;
                if (p === 1) return filters.period1;
                if (p === 2) return filters.period2;
                if (p === 3) return filters.period3;
                if (p === 4) return filters.period4;
                if (p === 5) return filters.period5;
                return filters.period6plus;
            };

            manifoldState.fixedPoints.filter(isFpVisible).forEach(fp => {
                const isRenderedByOrbit = periodicState.isReady
                    && manifoldState.showOrbits
                    && (periodicState.orbits || []).some(orbit =>
                        isFpVisible({ ...fp, period: orbit.period }) &&
                        (orbit.points || []).some(([ox, oy]) => Math.hypot(ox - fp.x, oy - fp.y) < 1e-5)
                    );
                if (isRenderedByOrbit) return;

                const stabLower = (fp.stability || '').toLowerCase();
                const isAttractor = stabLower === 'attractor' || stabLower === 'stable';
                const isDualRepeller = stabLower === 'dualrepeller' || stabLower === 'dual_repeller' || stabLower === 'dual repeller';
                const isRepeller = !isDualRepeller && (stabLower === 'repeller' || stabLower === 'unstable');
                const isSaddle = stabLower === 'saddle';
                const color = isAttractor ? ORBIT_COLORS.attractor :
                    isDualRepeller ? ORBIT_COLORS.dualRepeller :
                    isRepeller ? ORBIT_COLORS.repeller :
                    isSaddle ? ORBIT_COLORS.saddlePoint : ORBIT_COLORS.periodicBlue;

                const fpRadius = radius * (isAttractor ? 1.15 : (isRepeller || isDualRepeller) ? 1.1 : 1.0);

                // Dark high-contrast outer ring so fixed point is clearly separated from manifold lines
                const ringGeom = new THREE.RingGeometry(fpRadius * 0.9, fpRadius * 1.35, 32);
                const ringMat = new THREE.MeshBasicMaterial({
                    color: new THREE.Color('#000000'),
                    transparent: true,
                    opacity: 0.95,
                    depthTest: false,
                    depthWrite: false,
                    side: THREE.DoubleSide
                });
                const ring = new THREE.Mesh(ringGeom, ringMat);
                ring.position.set(fp.x, fp.y, 0.49);
                ring.renderOrder = 99;
                ring.userData = { type: 'fixedPoint' };
                scene.add(ring);

                const geom = new THREE.CircleGeometry(fpRadius, 32);
                const mat = new THREE.MeshBasicMaterial({
                    color: new THREE.Color(color),
                    transparent: true,
                    opacity: 1.0,
                    depthTest: false,
                    depthWrite: false,
                    side: THREE.DoubleSide
                });
                const sphere = new THREE.Mesh(geom, mat);
                sphere.position.set(fp.x, fp.y, 0.5);
                sphere.renderOrder = 100;
                sphere.userData = {
                    type: 'fixedPoint',
                    period: fp.period || 1,
                    stability: fp.stability,
                    pos: { x: fp.x, y: fp.y },
                    normal: Number.isFinite(fp.nx) && Number.isFinite(fp.ny)
                        ? { x: fp.nx, y: fp.ny }
                        : null,
                    eigenvalues: fp.eigenvalues || null
                };
                scene.add(sphere);
            });
        }

        if (manifoldState.showTrail && manifoldState.trajectoryPoints.length > 0) {
            if (dynamicSystem === 'duffing_ode') {
                const points = manifoldState.trajectoryPoints.map(p => new THREE.Vector3(p.x, p.y, 0.25));
                const geom = new THREE.BufferGeometry().setFromPoints(points);
                const mat = new THREE.LineBasicMaterial({ color: new THREE.Color(ORBIT_COLORS.manifold), linewidth: 2, transparent: true, opacity: 0.8 });
                const line = new THREE.Line(geom, mat);
                line.userData.type = 'trajectory';
                scene.add(line);
            } else {
                manifoldState.trajectoryPoints.forEach((point, idx) => {
                    const normalizedIdx = idx / manifoldState.trajectoryPoints.length;
                    const size = Math.max(1e-6, (4.5 * (0.4 + 0.6 * normalizedIdx)) / pxPerUnit);
                    const geom = new THREE.CircleGeometry(size, 16);
                    const mat = new THREE.MeshBasicMaterial({
                        color: new THREE.Color(ORBIT_COLORS.trajectory),
                        opacity: 0.4 + 0.6 * normalizedIdx,
                        transparent: true,
                        depthTest: false,
                        depthWrite: false,
                        side: THREE.DoubleSide
                    });
                    const sphere = new THREE.Mesh(geom, mat);
                    sphere.position.set(point.x, point.y, 0.45);
                    sphere.renderOrder = 40;
                    sphere.userData.type = 'trajectory';
                    scene.add(sphere);
                });
            }
        }

        if (manifoldState.hasStarted && manifoldState.currentPoint) {
            const glowRadius = Math.max(1e-6, 7.5 / pxPerUnit);
            const glowGeom = new THREE.RingGeometry(glowRadius * 0.8, glowRadius * 1.2, 32);
            const currentColor = dynamicSystem === 'duffing_ode' ? ORBIT_COLORS.manifold : ORBIT_COLORS.trajectory;
            const glowMat = new THREE.MeshBasicMaterial({
                color: new THREE.Color(currentColor),
                opacity: 0.6,
                transparent: true,
                side: THREE.DoubleSide,
                depthTest: false,
                depthWrite: false
            });
            const glowRing = new THREE.Mesh(glowGeom, glowMat);
            glowRing.position.set(manifoldState.currentPoint.x, manifoldState.currentPoint.y, 0.51);
            glowRing.renderOrder = 101;
            glowRing.userData.type = 'trajectory';
            scene.add(glowRing);

            const geom = new THREE.CircleGeometry(Math.max(1e-6, 5.0 / pxPerUnit), 32);
            const mat = new THREE.MeshBasicMaterial({
                color: new THREE.Color('#ffffff'),
                transparent: true,
                opacity: 1.0,
                depthTest: false,
                depthWrite: false,
                side: THREE.DoubleSide
            });
            const sphere = new THREE.Mesh(geom, mat);
            sphere.position.set(manifoldState.currentPoint.x, manifoldState.currentPoint.y, 0.52);
            sphere.renderOrder = 102;
            sphere.userData.type = 'trajectory';
            scene.add(sphere);
        }

    }, [manifoldState, geometricOffsetState, invariantSetState, bdeState, dynamicSystem, type, viewRange, viewportRange, readViewportSize, boundaryLayers, hasBoundarySamples, hasVerifiedBoundaryCycles, params.epsilon, filters, periodicState.isReady, periodicState.orbits]);

    useEffect(() => {
        if (!sceneRef.current) return;
        if (!periodicState.isReady && manifoldState.showOrbits) {
            // Keep the last successfully rendered periodic points on screen
            // while the worker searches at the new parameter value.
            return;
        }

        const scene = sceneRef.current;
        const previousOrbitObjects: THREE.Mesh[] = [];
        scene.traverse(child => {
            if (child instanceof THREE.Mesh && child.userData.type === 'orbit') {
                previousOrbitObjects.push(child);
            }
        });
        previousOrbitObjects.forEach(object => {
            object.geometry?.dispose();
            if (Array.isArray(object.material)) {
                object.material.forEach(material => material.dispose());
            } else {
                object.material.dispose();
            }
            scene.remove(object);
        });

        if (!periodicState.isReady) return;

        const resultRevision = periodicState.resultRevision;
        let renderedPointCount = 0;
        const isVisible = (orbit: PeriodicOrbit): boolean => {
            if (orbit.period === 1) return filters.period1;
            if (orbit.period === 2) return filters.period2;
            if (orbit.period === 3) return filters.period3;
            if (orbit.period === 4) return filters.period4;
            if (orbit.period === 5) return filters.period5;
            return filters.period6plus;
        };
        const colorForOrbit = (orbit: PeriodicOrbit): string => {
            const stability = (orbit.stability || '').toLowerCase();
            if (stability === 'stable' || stability === 'attractor') return ORBIT_COLORS.attractor;
            if (stability === 'dualrepeller' || stability === 'dual_repeller' || stability === 'dual repeller') return ORBIT_COLORS.dualRepeller;
            if (stability === 'saddle') return ORBIT_COLORS.saddlePoint;
            if (stability === 'unstable' || stability === 'repeller') return ORBIT_COLORS.repeller;
            return ORBIT_COLORS.periodicBlue;
        };

        const ySpan = Math.abs(viewportRange.yMax - viewportRange.yMin);
        const { height: vpHeight } = readViewportSize();
        const frustumH = Math.max(1e-6, ySpan + 0.24);
        const pxPerUnit = vpHeight / frustumH;
        const orbitSphereRadius = Math.max(1e-6, 5.5 / pxPerUnit);

        if (manifoldState.showOrbits) {
            periodicState.orbits.filter(isVisible).forEach((orbit, orbitIndex) => {
                const orbitId = `orbit-${orbit.period}-${orbitIndex}`;
                const pointColor = colorForOrbit(orbit);
                orbitExtendedStates(orbit).forEach((extendedPoint, pointIndex) => {
                    // Dark high-contrast outer ring so orbit point is clearly visible on top of manifold lines
                    const ringGeom = new THREE.RingGeometry(orbitSphereRadius * 0.9, orbitSphereRadius * 1.35, 32);
                    const ringMat = new THREE.MeshBasicMaterial({
                        color: new THREE.Color('#000000'),
                        transparent: true,
                        opacity: 0.95,
                        depthTest: false,
                        depthWrite: false,
                        side: THREE.DoubleSide
                    });
                    const ring = new THREE.Mesh(ringGeom, ringMat);
                    ring.position.set(extendedPoint.x, extendedPoint.y, 0.49);
                    ring.renderOrder = 99;
                    ring.userData = { type: 'orbit' };
                    scene.add(ring);

                    const geometry = new THREE.CircleGeometry(orbitSphereRadius, 32);
                    const material = new THREE.MeshBasicMaterial({
                        color: new THREE.Color(pointColor),
                        transparent: true,
                        opacity: 1.0,
                        depthTest: false,
                        depthWrite: false,
                        side: THREE.DoubleSide
                    });
                    const sphere = new THREE.Mesh(geometry, material);
                    sphere.position.set(extendedPoint.x, extendedPoint.y, 0.5);
                    sphere.renderOrder = 100;
                    sphere.userData = {
                        type: 'orbit',
                        resultRevision,
                        orbitId,
                        period: orbit.period,
                        stability: orbit.stability,
                        pointIndex,
                        pos: { x: extendedPoint.x, y: extendedPoint.y },
                        normal: Number.isFinite(extendedPoint.nx) && Number.isFinite(extendedPoint.ny)
                            ? { x: extendedPoint.nx, y: extendedPoint.ny }
                            : null,
                        orbitPoints: orbit.points,
                        eigenvalues: orbit.eigenvalues || null
                    };
                    scene.add(sphere);
                    renderedPointCount += 1;
                });
            });
        }

        if (rendererRef.current && cameraRef.current) {
            rendererRef.current.render(scene, cameraRef.current);
        }

        // Acknowledge the revision only after a browser paint opportunity.
        // The animation controller requires this acknowledgement before it
        // can request the next parameter value.
        let acknowledgementFrame: number | null = null;
        const paintFrame = requestAnimationFrame(() => {
            acknowledgementFrame = requestAnimationFrame(() => {
                setPeriodicState(previous => {
                    if (!previous.isReady || previous.resultRevision !== resultRevision) {
                        return previous;
                    }
                    if (
                        previous.renderedRevision === resultRevision
                        && previous.renderedPointCount === renderedPointCount
                    ) {
                        return previous;
                    }
                    return {
                        ...previous,
                        renderedRevision: resultRevision,
                        renderedPointCount
                    };
                });
            });
        });

        return () => {
            cancelAnimationFrame(paintFrame);
            if (acknowledgementFrame !== null) {
                cancelAnimationFrame(acknowledgementFrame);
            }
        };
    }, [
        filters,
        manifoldState.showOrbits,
        periodicState.isReady,
        periodicState.orbits,
        periodicState.resultRevision,
        readViewportSize,
        viewportRange
    ]);

    const stepForwardManifold = useCallback(() => {
        if (manifoldState.isRunning || !wasmModule) return;
        const { x, y, nx, ny } = manifoldState.currentPoint;

        if (!isFinite(x) || !isFinite(y) || Math.abs(x) > 10 || Math.abs(y) > 10) {
            setManifoldState(prev => ({
                ...prev,
                currentPoint: { ...prev.startPoint },
                trajectoryPoints: [],
                iteration: 0,
                hasStarted: false
            }));
            return;
        }

        let nextPoint: ExtendedState;
        if (dynamicSystem === 'custom') {
            if (!appliedParamValidation.valid) return;
            const result = wasmModule.boundary_map_user_defined(
                x, y, nx, ny,
                activeAppliedCustomEquations.xEq, activeAppliedCustomEquations.yEq,
                appliedParamValidation.normalized,
                params.epsilon
            );
            nextPoint = result;
        } else if (dynamicSystem === 'custom_ode') {
            if (!appliedParamValidation.valid) return;
            const result = wasmModule.boundary_map_user_defined_ode(
                x, y, nx, ny,
                activeAppliedCustomEquations.xEq, activeAppliedCustomEquations.yEq,
                appliedParamValidation.normalized,
                params.h,
                params.epsilon
            );
            nextPoint = result;
        } else if (dynamicSystem === 'duffing_ode') {
            const { boundary_map_duffing_ode } = wasmModule;
            if (!boundary_map_duffing_ode) {
                console.error('boundary_map_duffing_ode not found in WASM module');
                return;
            }
            nextPoint = boundary_map_duffing_ode(x, y, nx, ny, params.delta, params.h, params.epsilon);
        } else {
            const { boundary_map } = wasmModule;
            if (!boundary_map) {
                console.error('boundary_map not found in WASM module');
                return;
            }
            nextPoint = boundary_map(x, y, nx, ny, params.a, params.b, params.epsilon);
        }

        const isContinuousStep = dynamicSystem === 'duffing_ode' || dynamicSystem === 'custom_ode';
        setManifoldState(prev => ({
            ...prev,
            currentPoint: { x: nextPoint.x, y: nextPoint.y, nx: nextPoint.nx, ny: nextPoint.ny },
            trajectoryPoints: appendTrajectoryHistoryPoint({
                points: prev.trajectoryPoints,
                point: { x, y, nx, ny },
                iteration: prev.iteration,
                isContinuous: isContinuousStep,
                maxHistory: 1000
            }),
            iteration: prev.iteration + 1,
            hasStarted: true
        }));
    }, [manifoldState.isRunning, manifoldState.currentPoint, wasmModule, params, dynamicSystem, activeAppliedCustomEquations, appliedParamValidation]);

    const runToConvergenceManifold = useCallback(() => {
        if (!wasmModule) return;

        if (manifoldState.isRunning) {
            if (batchAnimationRef.current !== null) {
                cancelAnimationFrame(batchAnimationRef.current);
            }
            setManifoldState(prev => ({ ...prev, isRunning: false }));
            return;
        }

        setManifoldState(prev => ({ ...prev, isRunning: true }));

        const stepFn = (
            cx: number,
            cy: number,
            cnx: number,
            cny: number,
        ): ExtendedState | null => {
            if (dynamicSystem === 'custom') {
                if (!appliedParamValidation.valid) return null;
                return wasmModule.boundary_map_user_defined(
                    cx, cy, cnx, cny,
                    activeAppliedCustomEquations.xEq, activeAppliedCustomEquations.yEq,
                    appliedParamValidation.normalized,
                    params.epsilon
                );
            } else if (dynamicSystem === 'custom_ode') {
                if (!appliedParamValidation.valid) return null;
                return wasmModule.boundary_map_user_defined_ode(
                    cx, cy, cnx, cny,
                    activeAppliedCustomEquations.xEq, activeAppliedCustomEquations.yEq,
                    appliedParamValidation.normalized,
                    params.h,
                    params.epsilon
                );
            } else if (dynamicSystem === 'duffing_ode') {
                const { boundary_map_duffing_ode } = wasmModule;
                if (!boundary_map_duffing_ode) return null;
                return boundary_map_duffing_ode(cx, cy, cnx, cny, params.delta, params.h, params.epsilon);
            } else {
                const { boundary_map } = wasmModule;
                if (!boundary_map) return null;
                return boundary_map(cx, cy, cnx, cny, params.a, params.b, params.epsilon);
            }
        };

        let currentX = manifoldState.currentPoint.x;
        let currentY = manifoldState.currentPoint.y;
        let currentNx = manifoldState.currentPoint.nx;
        let currentNy = manifoldState.currentPoint.ny;
        let iteration = manifoldState.iteration;
        const newPoints = [...manifoldState.trajectoryPoints];

        const isContinuous = dynamicSystem === 'duffing_ode' || dynamicSystem === 'custom_ode';
        const limitIterations = !isContinuous;
        const currentBatchSize = isContinuous ? 15 : 5;

        const animateStep = () => {
            for (let i = 0; i < currentBatchSize; i++) {
                if (limitIterations && iteration >= params.maxIterations) break;

                if (!isFinite(currentX) || !isFinite(currentY) || Math.abs(currentX) > 10 || Math.abs(currentY) > 10) {
                    setManifoldState(prev => ({
                        ...prev,
                        isRunning: false,
                        hasStarted: true,
                        trajectoryPoints: newPoints,
                        currentPoint: { x: currentX, y: currentY, nx: currentNx, ny: currentNy },
                        iteration
                    }));
                    return;
                }

                if (shouldRecordTrajectoryHistoryPoint({ isContinuous, iteration })) {
                    if (isContinuous && newPoints.length >= 1000) {
                        newPoints.shift();
                    }
                    newPoints.push({ x: currentX, y: currentY, nx: currentNx, ny: currentNy });
                }

                const next = stepFn(currentX, currentY, currentNx, currentNy);
                if (!next) {
                    setManifoldState(prev => ({ ...prev, isRunning: false }));
                    return;
                }
                currentX = next.x;
                currentY = next.y;
                currentNx = next.nx;
                currentNy = next.ny;
                iteration++;
            }

            setManifoldState(prev => ({
                ...prev,
                currentPoint: { x: currentX, y: currentY, nx: currentNx, ny: currentNy },
                trajectoryPoints: [...newPoints],
                iteration,
                hasStarted: true
            }));

            if (!limitIterations || iteration < params.maxIterations) {
                batchAnimationRef.current = requestAnimationFrame(animateStep);
            } else {
                setManifoldState(prev => ({ ...prev, isRunning: false }));
            }
        };
        batchAnimationRef.current = requestAnimationFrame(animateStep);
    }, [manifoldState, params, wasmModule, dynamicSystem, activeAppliedCustomEquations, appliedParamValidation]);

    const handleStepForwardManifold = useCallback(() => {
        if (applyIfNeededBeforeAction('step')) return;
        stepForwardManifold();
    }, [applyIfNeededBeforeAction, stepForwardManifold]);

    const handleRunToConvergenceManifold = useCallback(() => {
        if (manifoldState.isRunning) {
            runToConvergenceManifold();
            return;
        }
        if (applyIfNeededBeforeAction('play')) return;
        runToConvergenceManifold();
    }, [manifoldState.isRunning, applyIfNeededBeforeAction, runToConvergenceManifold]);

    useEffect(() => {
        const pendingAction = postApplyActionRef.current;
        if (!pendingAction) return;
        if (hasPendingInputChanges) return;
        if (manifoldState.isComputing) return;
        if (!periodicState.isReady) return;

        postApplyActionRef.current = null;
        if (pendingAction === 'play') {
            runToConvergenceManifold();
        } else if (pendingAction === 'step') {
            stepForwardManifold();
        }
    }, [hasPendingInputChanges, manifoldState.isComputing, periodicState.isReady, runToConvergenceManifold, stepForwardManifold]);

    const resetBdeFlow = useCallback(() => {
        if (bdeAnimRef.current) cancelAnimationFrame(bdeAnimRef.current);
        if (!wasmModule) return;
        if (bdeSimRef.current) {
            bdeSimRef.current.free();
        }
        if (dynamicSystem === 'custom_ode') {
            if (!appliedParamValidation.valid) return;
            bdeSimRef.current = new wasmModule.BdeSimulatorUserDefinedWasm(
                activeAppliedCustomEquations.xEq, activeAppliedCustomEquations.yEq,
                appliedParamValidation.normalized,
                params.epsilon,
                manifoldState.startPoint.x,
                manifoldState.startPoint.y,
                CONTINUOUS_BOUNDARY_FLOW_SETTINGS.initialRadius,
                CONTINUOUS_BOUNDARY_FLOW_SETTINGS.sampleCount
            );
        } else {
            bdeSimRef.current = new wasmModule.BdeSimulatorWasm(
                params.delta,
                params.epsilon,
                manifoldState.startPoint.x,
                manifoldState.startPoint.y,
                CONTINUOUS_BOUNDARY_FLOW_SETTINGS.initialRadius,
                CONTINUOUS_BOUNDARY_FLOW_SETTINGS.sampleCount
            );
        }
        const simulator = bdeSimRef.current;
        if (!simulator) return;
        setBdeState({
            points: simulator.get_points() as ExtendedState[],
            isRunning: false,
        });
    }, [wasmModule, params.delta, params.epsilon, manifoldState.startPoint.x, manifoldState.startPoint.y, dynamicSystem, activeAppliedCustomEquations, appliedParamValidation]);

    const resetManifold = useCallback(() => {
        if (batchAnimationRef.current) cancelAnimationFrame(batchAnimationRef.current);
        setManifoldState(prev => ({
            ...prev,
            currentPoint: { ...prev.startPoint },
            trajectoryPoints: [],
            iteration: 0,
            isRunning: false,
            hasStarted: false
        }));
    }, []);

    useEffect(() => {
        resetManifold();
    }, [params.a, params.b, params.delta, params.h, params.epsilon, activeAppliedCustomEquations, appliedParamValidation, dynamicSystem, resetManifold]);

    useEffect(() => {
        if (dynamicSystem === 'duffing_ode' || dynamicSystem === 'custom_ode') {
            resetBdeFlow();
        }
    }, [dynamicSystem, resetBdeFlow]);
    const computeUlam = useCallback(async () => {
        if (!wasmModule) return;
        if ((dynamicSystem === 'custom' || dynamicSystem === 'custom_ode') && !appliedParamValidation.valid) {
            setUlamState(prev => ({ ...prev, isComputing: false }));
            return;
        }
        setUlamState(prev => ({ ...prev, isComputing: true, needsRecompute: false }));
        try {
            const result = await runComputeTask('computeUlam', {
                dynamicSystem,
                params: {
                    a: params.a,
                    b: params.b,
                    delta: params.delta,
                    h: params.h
                },
                viewRange: {
                    xMin: viewRange.xMin,
                    xMax: viewRange.xMax,
                    yMin: viewRange.yMin,
                    yMax: viewRange.yMax
                },
                ulam: {
                    subdivisions: ulamState.subdivisions,
                    pointsPerBox: ulamState.pointsPerBox,
                    epsilon: ulamState.epsilon
                },
                customEquations: activeAppliedCustomEquations,
                customParams: appliedParamValidation.normalized,
                currentPoint: manifoldState.hasStarted && manifoldState.currentPoint
                    ? {
                        x: manifoldState.currentPoint.x,
                        y: manifoldState.currentPoint.y
                    }
                    : null
            });

            setUlamState(prev => ({
                ...prev,
                isComputing: false,
                gridBoxes: result?.boxes || [],
                invariantMeasure: result?.invariantMeasure || null,
                leftEigenvector: result?.leftEigenvector || null,
                currentBoxIndex: result?.currentBoxIndex ?? -1,
                selectedBoxIndex: null,
                transitions: null
            }));

        } catch (err) {
            console.error("Ulam computation failed:", err);
            setUlamState(prev => ({ ...prev, isComputing: false }));
        }
    }, [wasmModule, params.a, params.b, params.delta, params.h, ulamState.subdivisions, ulamState.pointsPerBox, ulamState.epsilon, manifoldState.hasStarted, manifoldState.currentPoint, dynamicSystem, activeAppliedCustomEquations, appliedParamValidation, viewRange, runComputeTask]);

    useEffect(() => {
        setUlamState(prev => ({ ...prev, epsilon: params.epsilon }));
    }, [params.epsilon]);

    useEffect(() => {
        if (!ulamState.showUlamOverlay || !wasmModule) return;

        if (ulamDebounceRef.current) {
            clearTimeout(ulamDebounceRef.current);
        }


        ulamDebounceRef.current = setTimeout(() => {
            computeUlam();
        }, 200);

        return () => {
            if (ulamDebounceRef.current) {
                clearTimeout(ulamDebounceRef.current);
            }
        };
    }, [
        ulamState.showUlamOverlay,
        wasmModule,
        params.a,
        params.b,
        params.delta,
        params.h,
        ulamState.epsilon,
        ulamState.subdivisions,
        ulamState.pointsPerBox,
        computeRequestId,
        computeUlam
    ]);

    useEffect(() => {
        if (!ulamState.showUlamOverlay || !ulamState.gridBoxes.length) return;

        if (manifoldState.hasStarted && manifoldState.currentPoint) {
            const boxIdx = getGridBoxIndex(
                manifoldState.currentPoint.x,
                manifoldState.currentPoint.y,
                viewRange,
                ulamState.subdivisions
            );

            if (boxIdx !== ulamState.currentBoxIndex) {
                setUlamState(prev => ({
                    ...prev,
                    currentBoxIndex: boxIdx,
                    transitions: prev.showCurrentBox ? null : prev.transitions,
                    selectedBoxIndex: prev.showCurrentBox ? boxIdx : prev.selectedBoxIndex
                }));
                if (ulamState.showCurrentBox && boxIdx >= 0) {
                    requestUlamTransitions(boxIdx, 'current');
                }
            }
        }
    }, [manifoldState.currentPoint, manifoldState.hasStarted, ulamState.showUlamOverlay, ulamState.gridBoxes.length, ulamState.subdivisions, ulamState.currentBoxIndex, ulamState.showCurrentBox, viewRange, requestUlamTransitions]);


    useEffect(() => {
        const scene = sceneRef.current;
        if (!scene) return;

        const cleanup = () => {
            const oldMesh = scene.getObjectByName('ulam-grid');
            if (oldMesh instanceof THREE.InstancedMesh) {
                oldMesh.geometry.dispose();
                oldMesh.material.dispose();
                scene.remove(oldMesh);
            }
        };

        if (!ulamState.showUlamOverlay || !ulamState.gridBoxes.length) {
            cleanup();
            return;
        }

        cleanup();

        const boxes = ulamState.gridBoxes;
        const count = boxes.length;
        const geometry = new THREE.PlaneGeometry(1, 1);
        const material = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.5,
            side: THREE.DoubleSide,
            depthWrite: false
        });

        const mesh = new THREE.InstancedMesh(geometry, material, count);
        mesh.name = 'ulam-grid';
        mesh.userData.type = 'ulamGrid';

        const dummy = new THREE.Object3D();
        const color = new THREE.Color();

        const transitionMap = new Map<number, number>();
        if (ulamState.selectedBoxIndex !== null && ulamState.transitions) {
            ulamState.transitions.forEach(t => {
                transitionMap.set(t.index, t.probability);
            });
        }

        let maxMeasure = 0;
        if (ulamState.invariantMeasure) {
            maxMeasure = Math.max(...ulamState.invariantMeasure);
        }

        boxes.forEach((box, i) => {
            const cx = box.center[0];
            const cy = box.center[1];
            const rx = box.radius[0];
            const ry = box.radius[1];

            dummy.position.set(cx, cy, -0.05);
            dummy.scale.set(rx * 2 * 0.95, ry * 2 * 0.95, 1);
            dummy.updateMatrix();
            mesh.setMatrixAt(i, dummy.matrix);

            const isCurrentBox = ulamState.showCurrentBox && i === ulamState.currentBoxIndex;
            const isSelectedBox = ulamState.selectedBoxIndex !== null && i === ulamState.selectedBoxIndex;

            if (isCurrentBox && !isSelectedBox) {
                color.setHex(0xff00ff);
                mesh.setColorAt(i, color);
            } else if (ulamState.selectedBoxIndex !== null) {
                if (i === ulamState.selectedBoxIndex) {
                    color.setHex(0x00ffff);
                    mesh.setColorAt(i, color);
                } else if (transitionMap.has(i)) {
                    const prob = transitionMap.get(i) ?? 0;
                    color.setHSL(0.7 - prob * 0.7, 1.0, 0.5);
                    mesh.setColorAt(i, color);
                } else {
                    color.setHex(0x222222);
                    mesh.setColorAt(i, color);
                }
            } else if (ulamState.invariantMeasure && ulamState.invariantMeasure.length === count) {
                const measure = ulamState.invariantMeasure[i];
                if (measure > 0) {
                    const intensity = measure / maxMeasure;
                    const h = 0.66 - (intensity * 0.5);
                    color.setHSL(h, 1.0, 0.5);
                    mesh.setColorAt(i, color);
                } else {
                    color.setHex(0x111111);
                    mesh.setColorAt(i, color);
                }
            } else {
                color.setHex(0x333333);
                mesh.setColorAt(i, color);
            }
        });

        mesh.instanceMatrix.needsUpdate = true;
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

        scene.add(mesh);

        return cleanup;
    }, [ulamState.showUlamOverlay, ulamState.gridBoxes, ulamState.selectedBoxIndex, ulamState.transitions, ulamState.invariantMeasure, ulamState.currentBoxIndex, ulamState.showCurrentBox]);

    const setType = (newType: SystemType): void => {
        if (newType === 'continuous') setDynamicSystem('duffing_ode');
        if (newType === 'discrete') setDynamicSystem('henon');
    };

    const updateStartPoint = (extendedPoint: ExtendedState): void => {
        setManifoldState(prev => {
            return applyStartPointUpdate(prev, extendedPoint);
        });
        if (typeof window.update_start_point === 'function') {
            window.update_start_point(
                extendedPoint.x,
                extendedPoint.y,
                extendedPoint.nx,
                extendedPoint.ny
            );
        }
    };

    const applyPreset = (presetVals: Partial<BistParameters>): void => {
        setDraftParams(prev => ({ ...prev, ...presetVals }));
    };

    const exportExperiment = useCallback(() => {
        try {
            const selectedContour = geometricOffsetState.contours.find(contour => (
                contour.id === geometricOffsetState.selectedContourId
            )) ?? geometricOffsetState.contours[0];
            const inversePositionTolerance = selectedContour?.inverseResult
                ? inverseOffsetPositionTolerance(
                    geometricOffsetSampleSpacing(selectedContour.result)
                )
                : null;
            const bundle = buildExperimentBundle({
                commit: import.meta.env.VITE_GIT_COMMIT || 'development',
                configuration: {
                    dynamicSystem,
                    params,
                    customEquations,
                    customParams,
                    viewRange,
                    viewportRange,
                    periodicSearchSettings,
                    startPoint: manifoldState.startPoint,
                    solverSettings: {
                        manifold: {
                            intersectionThreshold: manifoldState.intersectionThreshold,
                            maximumPointSpacing: manifoldState.maximumManifoldPointSpacing,
                            computeStable: manifoldState.showStableManifold,
                            computeUnstable: manifoldState.showUnstableManifold
                        },
                        continuousBoundaryFlow: {
                            ...CONTINUOUS_BOUNDARY_FLOW_SETTINGS,
                            stepSize: params.h
                        },
                        ulam: {
                            subdivisions: ulamState.subdivisions,
                            pointsPerBox: ulamState.pointsPerBox,
                            epsilon: ulamState.epsilon,
                            integrationTime: type === 'continuous'
                                ? continuousUlamIntegrationTime(params.h)
                                : null,
                            ...ULAM_OPERATOR_SETTINGS
                        },
                        geometricOffsets: {
                            ...DEFAULT_GEOMETRIC_OFFSET_SETTINGS,
                            contourEpsilon: selectedContour?.epsilon
                                ?? DEFAULT_GEOMETRIC_OFFSET_SETTINGS.contourEpsilon,
                            contourEpsilons: geometricOffsetState.contours.map(contour => contour.epsilon),
                            inverseIterations: geometricOffsetState.inverseIterations,
                            inversePositionTolerance,
                            inversePositionToleranceRule: INVERSE_OFFSET_POSITION_TOLERANCE_RULE
                        }
                    }
                },
                results: {
                    periodicOrbits: periodicState.orbits,
                    periodicComputeMethod: periodicState.computeMethod,
                    periodicSupport: ulamSupportRef.current,
                    manifolds: manifoldState.manifolds,
                    stableManifolds: manifoldState.stableManifolds,
                    fixedPoints: manifoldState.fixedPoints,
                    intersections: manifoldState.intersections,
                    continuousBoundaryPoints: bdeState.points,
                    geometricOffsets: {
                        batchVersion: 1,
                        selectedContourId: geometricOffsetState.selectedContourId,
                        contours: geometricOffsetState.contours
                            .filter(contour => contour.result !== null)
                            .map(contour => ({
                                id: contour.id,
                                epsilon: contour.epsilon,
                                visible: contour.visible,
                                result: contour.result,
                            })),
                    },
                    inverseGeometricOffsets: {
                        batchVersion: 1,
                        sourceIds: geometricOffsetState.preimageSourceIds,
                        contours: geometricOffsetState.contours
                            .filter(contour => contour.inverseResult !== null)
                            .map(contour => ({
                                id: contour.id,
                                epsilon: contour.epsilon,
                                result: contour.inverseResult,
                            })),
                    },
                    ulam: {
                        gridBoxes: ulamState.gridBoxes,
                        invariantMeasure: ulamState.invariantMeasure,
                        leftEigenvector: ulamState.leftEigenvector
                    },
                    parameterSweep: sweepState.results
                }
            });
            const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const anchor = document.createElement('a');
            anchor.href = url;
            anchor.download = `bist-${dynamicSystem}-${new Date().toISOString().slice(0, 10)}.json`;
            document.body.appendChild(anchor);
            anchor.click();
            document.body.removeChild(anchor);
            URL.revokeObjectURL(url);
            setExperimentStatus({ type: 'success', message: 'Experiment exported with configuration and provenance.' });
        } catch (error) {
            setExperimentStatus({
                type: 'error',
                message: `Export failed: ${error instanceof Error ? error.message : String(error)}`,
            });
        }
    }, [
        customEquations,
        customParams,
        bdeState.points,
        dynamicSystem,
        geometricOffsetState.contours,
        geometricOffsetState.inverseIterations,
        geometricOffsetState.preimageSourceIds,
        geometricOffsetState.selectedContourId,
        manifoldState.fixedPoints,
        manifoldState.intersectionThreshold,
        manifoldState.maximumManifoldPointSpacing,
        manifoldState.intersections,
        manifoldState.manifolds,
        manifoldState.showStableManifold,
        manifoldState.showUnstableManifold,
        manifoldState.stableManifolds,
        manifoldState.startPoint,
        params,
        periodicSearchSettings,
        periodicState.computeMethod,
        periodicState.orbits,
        sweepState.results,
        type,
        ulamState.epsilon,
        ulamState.gridBoxes,
        ulamState.invariantMeasure,
        ulamState.leftEigenvector,
        ulamState.pointsPerBox,
        ulamState.subdivisions,
        viewRange,
        viewportRange
    ]);

    const importExperiment = useCallback(async (text: string) => {
        try {
            const bundle = parseExperimentBundle(text);
            const configuration = experimentConfigurationToUiState(bundle.configuration);
            const nextEquations = structuredClone(configuration.customEquations);
            const nextCustomParams = structuredClone(configuration.customParams);
            const nextParams = { ...configuration.params };
            const nextSearchSettings = { ...configuration.periodicSearchSettings };
            const nextViewRange = normalizeViewRange(configuration.viewRange);
            const nextViewportRange = constrainViewRange(
                configuration.viewportRange,
                nextViewRange,
            );

            setDynamicSystem(configuration.dynamicSystem);
            setParams(nextParams);
            setDraftParams(nextParams);
            setCustomEquations(nextEquations);
            setDraftCustomEquations(structuredClone(nextEquations));
            setCustomParams(nextCustomParams);
            setDraftCustomParams(structuredClone(nextCustomParams));
            setPeriodicSearchSettings(nextSearchSettings);
            setDraftPeriodicSearchSettings(nextSearchSettings);
            computationViewRangeRef.current = nextViewRange;
            viewportRangeRef.current = nextViewportRange;
            viewportRangeTargetRef.current = nextViewportRange;
            setViewRange(nextViewRange);
            setViewportRange(nextViewportRange);
            setEquationError(null);
            setPeriodicState(prev => ({
                ...prev,
                orbits: [],
                isReady: false,
                computeMethod: null,
                resultRevision: prev.resultRevision + 1
            }));
            setManifoldState(prev => ({
                ...prev,
                manifolds: [],
                rawManifolds: [],
                stableManifolds: [],
                fixedPoints: [],
                intersections: [],
                startPoint: configuration.startPoint,
                currentPoint: configuration.startPoint,
                trajectoryPoints: [],
                iteration: 0,
                isRunning: false,
                hasStarted: false,
                intersectionThreshold: configuration.manifoldSettings.intersectionThreshold,
                maximumManifoldPointSpacing: configuration.manifoldSettings.maximumPointSpacing,
                showStableManifold: configuration.manifoldSettings.computeStable,
                showUnstableManifold: configuration.manifoldSettings.computeUnstable
            }));
            setGeometricOffsetState(prev => {
                const contours = reconcileGeometricOffsetContours(
                    [],
                    configuration.geometricOffsetSettings.contourEpsilons,
                );
                const selectedContour = contours.find(contour => (
                    contour.epsilon === configuration.geometricOffsetSettings.contourEpsilon
                )) ?? contours[0];
                return {
                    ...prev,
                    seriesStart: contours[0].epsilon,
                    seriesEnd: contours[contours.length - 1].epsilon,
                    seriesCount: contours.length,
                    individualEpsilon: configuration.geometricOffsetSettings.contourEpsilon,
                    contours,
                    selectedContourId: selectedContour.id,
                    preimageSourceIds: [selectedContour.id],
                    inverseIterations: configuration.geometricOffsetSettings.inverseIterations,
                    showInverseContours: true,
                    isComputing: false,
                    isComputingInverse: false,
                    error: null,
                    inverseError: null,
                };
            });
            setUlamState(prev => ({
                ...prev,
                subdivisions: configuration.ulamSettings.subdivisions,
                pointsPerBox: configuration.ulamSettings.pointsPerBox,
                epsilon: configuration.ulamSettings.epsilon,
                gridBoxes: [],
                invariantMeasure: null,
                leftEigenvector: null,
                transitions: null,
                selectedBoxIndex: null,
                currentBoxIndex: -1,
                needsRecompute: true
            }));
            setComputeRequestId(prev => prev + 1);
            setExperimentStatus({
                type: 'success',
                message: `Loaded BIST ${bundle.provenance.software.version} experiment schema v${bundle.schemaVersion}; results are being recomputed.`
            });
        } catch (error) {
            setExperimentStatus({
                type: 'error',
                message: `Import failed: ${error instanceof Error ? error.message : String(error)}`,
            });
        }
    }, []);

    return (
        <Shell>
            <Sidebar
                type={type}
                setType={setType}
                dynamicSystem={dynamicSystem}
                setDynamicSystem={setDynamicSystem}
                SYSTEMS={SYSTEM_CATALOG}
                exportExperiment={exportExperiment}
                importExperiment={importExperiment}
                experimentStatus={experimentStatus}
                customEquations={draftCustomEquations}
                setCustomEquations={setDraftCustomEquations}
                equationError={equationError}
                params={draftParams}
                setParams={setDraftParams}
                applyPreset={applyPreset}
                customParams={activeDraftCustomParams}
                setCustomParams={(next) => {
                    setDraftCustomParams(prev => ({
                        ...prev,
                        [activeCustomKey]: typeof next === 'function' ? next(prev[activeCustomKey]) : next
                    }));
                }}
                paramErrors={draftParamValidation.errors}
                hasPendingInputChanges={hasPendingInputChanges}
                applyInputsAndRecompute={applyInputsAndRecompute}
                appliedParams={params}
                viewRange={viewRange}
                setViewRange={updateViewRange}
                rangeLimit={RANGE_LIMIT}
                resetViewRange={resetViewRange}
                manifoldState={manifoldState}
                setManifoldState={setManifoldState}
                geometricOffsetState={geometricOffsetState}
                setGeometricOffsetState={setGeometricOffsetState}
                invariantSetState={invariantSetState}
                setInvariantSetState={setInvariantSetState}
                computeForwardInvariantSet={computeForwardInvariantSet}
                hasBoundarySamples={hasBoundarySamples}
                boundaryLayerError={boundaryLayers.error}
                boundarySampling={{
                    unstable: boundaryLayers.unstableSampling,
                    deterministic: boundaryLayers.deterministicSampling,
                }}
                canComputeGeometricOffsets={!manifoldState.isComputing && calculatedBoundaryBranches.length > 0}
                computeGeometricOffsets={computeGeometricOffsets}
                computeInverseGeometricOffsets={computeInverseGeometricOffsets}
                fitInverseGeometricOffsets={fitInverseGeometricOffsets}
                canComputeInverseGeometricOffsets={geometricOffsetSourceContours(geometricOffsetState).length > 0
                    && !geometricOffsetState.isComputing
                    && !geometricOffsetState.isComputingInverse}
                ORBIT_COLORS={ORBIT_COLORS}
                filters={filters}
                setFilters={setFilters}
                periodicState={periodicState}
                periodicSearchSettings={draftPeriodicSearchSettings}
                appliedPeriodicSearchSettings={periodicSearchSettings}
                updatePeriodicSearchSettings={updatePeriodicSearchSettings}
                runPeriodicGridSearch={runPeriodicGridSearch}
                updateStartPoint={updateStartPoint}
                animationState={animationState}
                setAnimationState={setAnimationState}
                recordingState={recordingState}
                startAnimation={startAnimation}
                stopAnimation={stopAnimation}
                toggleRecording={toggleRecording}
                ulamState={ulamState}
                setUlamState={setUlamState}
                wasmModule={wasmModule}
                sweepState={sweepState}
                setSweepState={setSweepState}
                bdeState={bdeState}
                stepForwardManifold={handleStepForwardManifold}
                runToConvergenceManifold={handleRunToConvergenceManifold}
                resetManifold={resetManifold}
                resetBdeFlow={resetBdeFlow}
            />
            <Viewport
                type={type}
                canvasRef={canvasRef}
                tooltip={tooltip}
                manifoldState={manifoldState}
                geometricOffsetState={geometricOffsetState}
                invariantSetState={invariantSetState}
                ulamState={ulamState}
                hasBoundarySamples={hasBoundarySamples}
                displayRange={viewportRange}
                savePNG={savePNG}
                handleZoomIn={handleZoomIn}
                handleZoomOut={handleZoomOut}
                handleResetView={resetViewportRange}
                handlePanMode={handlePanMode}
                isPanMode={isPanMode}
            />
        </Shell>
    );
}

export default SetValuedViz;
