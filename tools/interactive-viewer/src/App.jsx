// Copyright 2025 Intrinsic Innovation LLC

// ---------------------------------------------------------------------------
// App.jsx
// ---------------------------------------------------------------------------
// Interactive Mesh Simplification – a web app for interactive mesh simplification
// This app allows users to upload 3D models, simplify them, and visualize the results.
// -----------------------------------------------------------------------------

import React, {
  Suspense,
  useEffect,
  useRef,
  useState,
  useCallback,
} from "react";

// Three.js core and submodules
import * as THREE from "three";
import { ConvexGeometry } from "three/examples/jsm/geometries/ConvexGeometry.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader";

// @react-three/fiber and drei
import { Canvas, useLoader } from "@react-three/fiber";
import {
  Environment,
  GizmoHelper,
  GizmoViewport,
  Html,
  OrbitControls,
  Stats,
} from "@react-three/drei";

// Local components
import BottomLeftMenu from "./components/BottomLeftMenu";
import ColorBar from "./components/ColorBar";
import FileLoader from "./components/FileLoader";
import SliderWithButtons from "./components/SliderWithButtons";
import StatsPanel from "./components/StatsPanel";
import TopRightControls from "./components/TopRightControls";
import FaceHandles from "./components/box/FaceHandles";
import BoxSelectControls from "./components/box/BoxSelectControls";
import FaceSnapControls from "./components/box/FaceSnapControls";
import HullItem from "./components/HullItem";

// Local utilities
import { computeMergedHullVolume, pointsFromMesh } from "./utils/volume";
import useMeshes from "./hooks/useMeshes";
import useBoundingBoxes from "./hooks/useBoundingBoxes";
import useOriginalModel from "./hooks/useOriginalModel";
import useSceneIO from "./hooks/useSceneIO";

// Side-effects (styles/tests)
import "./utils/e2e-test";

import useKeyboardShortcuts from "./hooks/useKeyboardShortcuts";
import useBackend from "./hooks/useBackend";

function MainMesh({ url, wireframe }) {
  const gltf = useLoader(GLTFLoader, url);

  // Mark meshes + update wireframe flag whenever it changes
  useEffect(() => {
    gltf.scene?.traverse((c) => {
      if (c.isMesh) {
        c.userData.isModelMesh = true;
        c.material.wireframe = wireframe;
      }
    });
  }, [gltf, wireframe]);

  return (
    <group>
      <primitive object={gltf.scene} />
    </group>
  );
}

// Traverse each loaded mesh object and assign it a stable, unique name like
// "hull-nonSelect-0" or "hull-select-1". This helps ensure consistent identification
// of meshes during selection, hiding, and other interactions across re-renders.
function ProcessedMeshes({
  nonSelectUrl,
  selectUrls = [],
  nonSelectErrUrlF,
  selectErrUrlsF = [],
  nonSelectErrUrlR,
  selectErrUrlsR = [],
  showApproximation,
  showErrorForward,
  showErrorReverse,
  wireframe,
  onSelectHull,
  selectedHulls,
  hullVolumes,
  hiddenHulls,
  onPickForHide,
}) {
  const meshUrls =
    showApproximation && nonSelectUrl
      ? [...(nonSelectUrl || []), ...(selectUrls || [])]
      : [];

  const forwardErrUrls =
    showErrorForward && nonSelectErrUrlF
      ? [nonSelectErrUrlF, ...selectErrUrlsF]
      : [];
  const reverseErrUrls =
    showErrorReverse && nonSelectErrUrlR
      ? [nonSelectErrUrlR, ...selectErrUrlsR]
      : [];

  const errorGeomsF = useLoader(PLYLoader, forwardErrUrls);
  const errorGeomsR = useLoader(PLYLoader, reverseErrUrls);

  const processedObjs = useLoader(GLTFLoader, meshUrls);

  /* give every mesh a stable name once */
  React.useEffect(() => {
    processedObjs.forEach((obj, i) => {
      const nonSelCount = nonSelectUrl?.length ?? 0;
      const group = i < nonSelCount ? "nonSelect" : "select";
      const localIdx = group === "nonSelect" ? i : i - nonSelCount;
      obj.scene.traverse(
        (c) => c.isMesh && (c.name = `hull-${group}-${localIdx}`),
      );
    });
  }, [processedObjs, nonSelectUrl]);

  return (
    <>
      {processedObjs.map((obj, i) => {
        const nonSelCount = nonSelectUrl?.length ?? 0;
        const group = i < nonSelCount ? "nonSelect" : "select";
        const localIdx = group === "nonSelect" ? i : i - nonSelCount;
        const meshKey = obj.scene.uuid;

        return (
          <HullItem
            key={meshKey}
            obj={obj}
            group={group}
            localIdx={localIdx}
            meshKey={meshKey}
            wireframe={wireframe}
            selectedHulls={selectedHulls}
            hullVolumes={hullVolumes}
            hiddenHulls={hiddenHulls}
            onSelectHull={onSelectHull}
            onPickForHide={onPickForHide}
          />
        );
      })}

      {errorGeomsF.map((g, i) => (
        <mesh key={`errF-${i}`} geometry={g}>
          <meshBasicMaterial vertexColors transparent opacity={1} />
        </mesh>
      ))}
      {errorGeomsR.map((g, i) => (
        <points key={`errR-${i}`} geometry={g}>
          <pointsMaterial
            vertexColors={true}
            size={0.002}
            sizeAttenuation={true}
            transparent={true}
            opacity={0.8}
          />
        </points>
      ))}
    </>
  );
}

export default function App() {
  // Which boxes should contribute convex hulls to error processing
  const [errorBoxFilter, setErrorBoxFilter] = useState(new Set()); // Set<number>
  // Map of box indices to hull indices, for both non-selected and selected boxes
  const [boxToHullMap, setBoxToHullMap] = useState({
    nonSelect: {},
    select: {},
  });

  const [showOriginal, setShowOriginal] = useState(true);
  const [showApproximation, setShowApproximation] = useState(false);
  const [showErrorForward, setShowErrorForward] = useState(false);
  const [showErrorReverse, setShowErrorReverse] = useState(false);
  const [errsForward, setErrsForward] = useState({
    nonSelectErr: null,
    selectErrs: [],
  });
  const [errsReverse, setErrsReverse] = useState({
    nonSelectErr: null,
    selectErrs: [],
  });

  const [isEditing, setIsEditing] = useState(true);

  const orbitRef = useRef();
  const [errorMin, setErrorMin] = useState(0);
  const [errorMax, setErrorMax] = useState(0.01);

  const [statsApprox, setStatsApprox] = useState({
    vertices: null,
    faces: null,
    hulls: null,
    rt_factor: null,
  });
  const [displayStats, setDisplayStats] = useState({
    vertices: "--",
    faces: "--",
    hulls: "--",
    rt_factor: "--",
  });

  const [meshLabels, setMeshLabels] = useState({
    nonSelectConvex: [],
    nonSelectBox: [],
    selectConvex: [],
    selectBox: [],
  });
  const [wireframeMode, setWireframeMode] = useState(false);
  const [timings, setTimings] = useState({
    process: null,
    runError: null,
    perf: null,
  });
  const sceneRef = useRef();
  const [allHulls, setAllHulls] = useState([]);
  const [sumVolume, setSumVolume] = useState(0);
  const [unionVolume, setUnionVolume] = useState(0);
  const [cost, setCost] = useState(0);
  const [unionGeom, setUnionGeom] = useState(null);

  // keep track of which hull (or “original”) the user last clicked:
  const [activeHideTarget, setActiveHideTarget] = useState(null);

  const [algorithm, setAlgorithm] = useState("vhacd"); // vhacd or coacd
  /* ——‑ simple aliases so the UI keeps its old prop names —— */

  const {
    /* state */
    boundingBoxes,
    setBoundingBoxes,
    defaultSize,
    setDefaultSize,
    defaultCoarseness,
    setDefaultCoarseness,
    showCanvasSliders,
    setShowCanvasSliders,

    showBoxes,
    setShowBoxes,
    boxMode,
    setBoxMode,
    selectedBoxIndex,
    setSelectedBoxIndex,
    hoverFace,
    setHoverFace,
    activeFace,
    setActiveFace,
    errors,
    setErrors,
    /* helpers */
    handleAddBox,
    updateBoxPosition,
    updateBoxSize,
    updateBoxCoarsenessValue,
    updateDefaultSizeValue,
    updateDefaultCoarsenessValue,
    handleDeleteBox,
    handleClear,
    handleSnapBoxes,
  } = useBoundingBoxes({ sceneRef });

  const {
    meshes,
    hullVolumes,
    selectedHulls,
    hiddenHulls,
    setMeshes,
    setHullVolumes,
    toggleHull,
    deleteHull,
    clearHullSelections,
    handleMergeHulls,
    setHiddenHulls,
    setSelectedHulls,
  } = useMeshes(sceneRef);
  const {
    originalUrl,
    setOriginalUrl,
    checkWatertight,
    handleWatertight,
    handleDecimate,
    countMeshStats,
    loadDefault,
    handleFileInput,
    isLoading,
    setIsLoading,
    statsOrig,
    setStatsOrig,
    meshDimensions,
    setMeshDimensions,
    watertight,
  } = useOriginalModel({ sceneRef });

  useKeyboardShortcuts({
    selectedHulls,
    clearHullSelections,
    deleteHull,
    selectedBoxIndex,
    handleDeleteBox,
    handleMergeHulls,
    setSelectedBoxIndex,
    setActiveFace,
    setHiddenHulls,
    setAllHulls,
  });
  const sceneIO = useSceneIO({
    // mesh / UI data it needs to save or restore
    meshes,
    meshLabels,
    boundingBoxes,
    hiddenHulls,
    showOriginal,
    showApproximation,
    showErrorForward,
    showErrorReverse,
    showBoxes,
    showCanvasSliders,
    wireframeMode,
    boxMode,
    isEditing,
    errorRange: { min: errorMin, max: errorMax },
    defaultSize,
    defaultCoarseness,
    meshDimensions,
    statsOrig,
    statsApprox,
    timings,
    selectedHulls,
    watertight,
    /* raw URLs + error sets */
    originalUrl,
    errsForward,
    errsReverse,
    /* setters for the import path */
    setOriginalUrl,
    checkWatertight,
    countMeshStats,
    setStatsOrig,
    setMeshes,
    setBoundingBoxes,
    setShowOriginal,
    setShowApproximation,
    setShowErrorForward,
    setShowErrorReverse,
    setShowBoxes,
    setShowCanvasSliders,
    setWireframeMode,
    setBoxMode,
    setIsEditing,
    setErrorRange: ({ min, max }) => {
      setErrorMin(min);
      setErrorMax(max);
    },
    setDefaultSize,
    setDefaultCoarseness,
    setMeshDimensions,
    setMeshLabels,
    setStatsApprox,
    setHiddenHulls,
    setSelectedHulls,
    setTimings,
    setErrsForward,
    setErrsReverse,
    setIsLoading,
    setBoxToHullMap,
    boxToHullMap,
  });

  const handleExportState = sceneIO.exportSceneState;
  const handleImportState = (file) => sceneIO.importSceneState(file);
  const handleExport = sceneIO.exportMeshesZip;

  const {
    decompose: handleSend, // ← POST /api/decompose
    runError: handleProcessError, // ← POST /api/run-error
    perfCheck: handlePerfCheck, // ← POST /api/mujoco-perf
  } = useBackend({
    /* read‑only inputs -------------------------------------------- */
    meshes, // { nonSelect: string[], select: string[] }
    boundingBoxes, // [{ center, size, coarseness }, ...]
    originalUrl, // blob URL of the true mesh
    errorBoxFilter, // Set<number> – boxes to include in error calc
    boxToHullMap, // { nonSelect: {boxIdx:[hullIdx]}, select:{...} }
    defaultCoarseness, // global fallback hull count
    algorithm, // 'vhacd' | 'coacd'
    errorRange: {
      // error colormap range
      min: errorMin,
      max: errorMax,
    },

    /* write‑back setters ------------------------------------------ */
    setMeshes,
    setHullVolumes,
    setStatsApprox,
    setBoxToHullMap,
    setMeshLabels,
    setErrsForward,
    setErrsReverse,
    setShowErrorForward,
    setShowErrorReverse,
    setTimings,
    setIsLoading,
    setShowApproximation,
    setShowOriginal,

    /* optional helper --------------------------------------------- */
    countMeshStatsForUrls, // (urls[]) → Promise<{ vertices, faces }>
  });

  useEffect(() => {
    if (selectedHulls.length < 2) {
      setSumVolume(0);
      setUnionVolume(0);
      setCost(0);
      setUnionGeom(null);
      return;
    }

    // 1) sum of the individual volumes:
    const sumV = selectedHulls.reduce(
      (acc, h) => acc + (h.stats.volume || 0),
      0,
    );

    // 2) look up the actual Mesh objects by the name you assigned:
    const meshes = selectedHulls
      .map((h) => sceneRef.current.getObjectByName(`hull-${h.group}-${h.idx}`))
      .filter((m) => m && m.isMesh);

    // 3) compute merged volume:
    const mergedV = computeMergedHullVolume(meshes);
    // 4) generate the ConvexGeometry for display:
    const allPts = meshes.flatMap(pointsFromMesh);
    const unionGeometry = new ConvexGeometry(
      allPts.map((p) => new THREE.Vector3(p[0], p[1], p[2])),
    );

    // 5) push to state:
    setSumVolume(sumV);
    setUnionVolume(mergedV);
    setCost(mergedV - sumV);
    setUnionGeom(unionGeometry);
  }, [selectedHulls]);
  useEffect(() => {
    // only run in e2e mode
    if (
      typeof window === "undefined" ||
      !window.location.search.includes("e2e")
    ) {
      return;
    }

    // build a clean JS array of plain objects
    const data = boundingBoxes.map((b, i) => ({
      index: i,
      center: {
        x: Number(b.center.x.toFixed(6)),
        y: Number(b.center.y.toFixed(6)),
        z: Number(b.center.z.toFixed(6)),
      },
      size: {
        x: Number(b.size.x.toFixed(6)),
        y: Number(b.size.y.toFixed(6)),
        z: Number(b.size.z.toFixed(6)),
      },
      coarseness: Number(b.coarseness.toFixed(6)),
    }));

    // one line: easy to split and parse
    console.e2e("[E2E] BOXES " + JSON.stringify(data));
  }, [boundingBoxes]);
  useEffect(() => {
    if (!originalUrl) return;
    const loader = new GLTFLoader();
    loader.load(originalUrl, (gltf) => {
      const bbox = new THREE.Box3().setFromObject(gltf.scene);
      const size = bbox.getSize(new THREE.Vector3());
      setMeshDimensions({ x: size.x, y: size.y, z: size.z });
    });
  }, [originalUrl]);
  useEffect(() => {
    if (!showBoxes && isEditing) {
      setIsEditing(false);
    }
    if (showBoxes) {
      setIsEditing(true);
    }
  }, [showBoxes, isEditing]);

  useEffect(() => {
    if (!showApproximation) return;

    // build the full list of URLs we’re rendering
    const urls = [...meshes.nonSelect, ...meshes.select];

    countMeshStatsForUrls(urls).then(({ vertices, faces }) => {
      if (window.location.search.includes("e2e")) {
        console.e2e(
          `[E2E] New stats → ` +
            `Vertices: ${vertices}, Faces: ${faces}, Hulls: ${
              meshes.nonSelect.length + meshes.select.length
            }`,
        );
      }
      setStatsApprox({
        vertices,
        faces,
        // total # of hull‐objects
        hulls: meshes.nonSelect.length + meshes.select.length,
        // leave rt_factor alone
        rt_factor: statsApprox.rt_factor,
      });
    });
  }, [meshes.nonSelect, meshes.select, showApproximation]);

  useEffect(() => {
    if (selectedBoxIndex == null) return; // nothing selected
    if (selectedBoxIndex >= boundingBoxes.length) {
      setSelectedBoxIndex(null); // deleted last box
      setActiveFace(null); // leave snap-mode
    }
  }, [boundingBoxes, selectedBoxIndex]);
  useEffect(() => {
    const fmt = (v) => (v == null ? "--" : v.toLocaleString());

    if (showOriginal && showApproximation) {
      setDisplayStats({
        vertices: `${fmt(statsOrig.vertices)} / ${fmt(statsApprox.vertices)}`,
        faces: `${fmt(statsOrig.faces)} / ${fmt(statsApprox.faces)}`,
        hulls: `-- / ${fmt(statsApprox.hulls)}`,
        rt_factor: `-- / ${fmt(statsApprox.rt_factor)}`,
      });
    } else if (showOriginal) {
      setDisplayStats({
        vertices: fmt(statsOrig.vertices),
        faces: fmt(statsOrig.faces),
        hulls: "--",
        rt_factor: "--",
      });
    } else if (showApproximation) {
      setDisplayStats({
        vertices: fmt(statsApprox.vertices),
        faces: fmt(statsApprox.faces),
        hulls: fmt(statsApprox.hulls),
        rt_factor: fmt(statsApprox.rt_factor),
      });
    } else {
      setDisplayStats({
        vertices: "--",
        faces: "--",
        hulls: "--",
        rt_factor: "--",
      });
    }
  }, [showOriginal, showApproximation, statsOrig, statsApprox]);

  const countMeshStatsForUrls = (urls) =>
    Promise.all(urls.map(countMeshStats)).then((arr) =>
      arr.reduce(
        (tot, s) => ({
          vertices: tot.vertices + (s.vertices ?? 0),
          faces: tot.faces + (s.faces ?? 0),
        }),
        { vertices: 0, faces: 0 },
      ),
    );

  const printCameraPose = useCallback(() => {
    const ctrl = orbitRef.current;
    if (!ctrl) return;
    const pos = ctrl.object.position.toArray().map((n) => +n.toFixed(4));
    const tgt = ctrl.target.toArray().map((n) => +n.toFixed(4));
    console.e2e(
      `[CAM] position = [${pos.join(", ")}], target = [${tgt.join(", ")}]`,
    );
  }, []);

  return (
    <>
      {window.location.search.includes("e2e") && (
        <button
          onClick={printCameraPose}
          style={{
            position: "absolute",
            top: 16,
            left: 200,
            zIndex: 2,
            padding: "4px 8px",
            background: "#222",
            color: "white",
            border: "none",
            borderRadius: 4,
            cursor: "pointer",
            fontSize: 12,
          }}
        >
          📸 Print Camera Pose
        </button>
      )}
      <StatsPanel stats={displayStats} timings={timings} />

      <TopRightControls
        errorBoxFilter={errorBoxFilter}
        setErrorBoxFilter={setErrorBoxFilter}
        boxToHullMap={boxToHullMap}
        showBoxes={showBoxes}
        setShowBoxes={setShowBoxes}
        wireframeMode={wireframeMode}
        setWireframeMode={setWireframeMode}
        boxMode={boxMode}
        setBoxMode={setBoxMode}
        isEditing={isEditing}
        setIsEditing={setIsEditing}
        handleSend={handleSend}
        handleProcessError={handleProcessError}
        meshes={meshes}
        boundingBoxes={boundingBoxes}
        handleSnapBoxes={handleSnapBoxes}
        handleExport={handleExport}
        handlePerfCheck={handlePerfCheck}
        isLoading={isLoading}
        showOriginal={showOriginal}
        setShowOriginal={setShowOriginal}
        showApproximation={showApproximation}
        setShowApproximation={setShowApproximation}
        showErrorForward={showErrorForward}
        setShowErrorForward={setShowErrorForward}
        showErrorReverse={showErrorReverse}
        setShowErrorReverse={setShowErrorReverse}
        handleExportState={handleExportState}
        handleImportState={handleImportState}
        handleDecimate={handleDecimate}
        handleWatertight={handleWatertight}
        originalUrl={originalUrl}
        watertight={watertight}
        algorithm={algorithm}
        setAlgorithm={setAlgorithm}
      />

      <Canvas
        camera={{ position: [-0.5428, 0.5733, 0.5202] }}
        gl={{
          toneMapping: THREE.ACESFilmicToneMapping,
          physicallyCorrectLights: false,
        }}
        onContextMenu={(e) => e.nativeEvent.preventDefault()}
        onCreated={({ gl, scene }) => {
          // ← get the renderer *and* the scene
          sceneRef.current = scene;
          const { width, height } = gl.getSize(new THREE.Vector2());
          console.e2e(`[E2E] viewport ${width} ${height}`);
          /* ───────── GPU / renderer info ───────── */
          // the raw WebGL context
          const ctx = gl.getContext();

          // extension needed for un-masked strings
          const dbg = ctx.getExtension("WEBGL_debug_renderer_info");

          // graceful fallback if the extension is not available
          const vendor = dbg
            ? ctx.getParameter(dbg.UNMASKED_VENDOR_WEBGL)
            : ctx.getParameter(ctx.VENDOR);
          const renderer = dbg
            ? ctx.getParameter(dbg.UNMASKED_RENDERER_WEBGL)
            : ctx.getParameter(ctx.RENDERER);
          const version = ctx.getParameter(ctx.VERSION);

          // very rough heuristic: SwiftShader / llvmpipe ⇒ CPU; anything else ⇒ GPU
          const isGPU = !/swiftshader|soft|llvmpipe/i.test(renderer);

          console.log("──────── WebGL info ────────");
          console.log("Vendor   :", vendor);
          console.log("Renderer :", renderer);
          console.log("Version  :", version);
          console.log("Using GPU:", isGPU);
        }}
      >
        <Suspense fallback={null}>
          <Environment preset="studio" />
          {<axesHelper args={[0.1]} />}

          {/* small widget that always sits in the corner */}
          <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
            <GizmoViewport
              axisColors={["#ff3653", "#8adb00", "#2f80ff"]}
              labelColor="white"
            />
          </GizmoHelper>

          <BoxSelectControls
            enabled={
              isEditing && activeFace == null && selectedBoxIndex == null
            }
            onAddBox={handleAddBox}
            orbitRef={orbitRef}
            mode={boxMode}
          />
          {showBoxes &&
            boundingBoxes.map((b, i) => {
              const isSelected = selectedBoxIndex === i;
              return (
                <React.Fragment key={i}>
                  <mesh
                    position={[b.center.x, b.center.y, b.center.z]}
                    scale={[b.size.x, b.size.y, b.size.z]}
                    userData={{ isHelper: true }}
                    // ← select on left‐click before anything else
                    onPointerDownCapture={(e) => {
                      e.stopPropagation();
                      e.nativeEvent.stopImmediatePropagation();
                      setSelectedBoxIndex(i);
                    }}
                    // still allow right‐click as before
                    onContextMenu={(e) => {
                      e.stopPropagation();
                      e.nativeEvent.preventDefault();
                      setSelectedBoxIndex(i);
                    }}
                  >
                    <boxGeometry args={[1, 1, 1]} />
                    <meshBasicMaterial
                      color={isSelected ? "#ff9900" : "yellow"}
                      wireframe
                    />
                  </mesh>

                  {isSelected && (
                    <FaceHandles
                      box={b}
                      boxIdx={i}
                      onHover={setHoverFace}
                      onPick={setActiveFace}
                      hover={hoverFace}
                      active={activeFace}
                      setSelectedBoxIndex={setSelectedBoxIndex}
                    />
                  )}
                </React.Fragment>
              );
            })}

          {meshes.nonSelect && (
            <ProcessedMeshes
              nonSelectUrl={meshes.nonSelect}
              selectUrls={meshes.select}
              nonSelectErrUrlF={errsForward.nonSelectErr}
              selectErrUrlsF={errsForward.selectErrs}
              // reverse errors
              nonSelectErrUrlR={errsReverse.nonSelectErr}
              selectErrUrlsR={errsReverse.selectErrs}
              showApproximation={showApproximation}
              showErrorForward={showErrorForward}
              showErrorReverse={showErrorReverse}
              wireframe={wireframeMode}
              onSelectHull={toggleHull}
              selectedHulls={selectedHulls}
              hullVolumes={hullVolumes}
              hiddenHulls={hiddenHulls}
              onPickForHide={(group, idx) =>
                setActiveHideTarget({ group, idx })
              }
            />
          )}
          {showOriginal && originalUrl /* draw LAST → on top */ && (
            <MainMesh url={originalUrl} wireframe={wireframeMode} />
          )}

          {showBoxes &&
            boundingBoxes.map((b, i) => {
              return (
                <React.Fragment key={i}>
                  <mesh
                    position={[b.center.x, b.center.y, b.center.z]}
                    userData={{ isHelper: true }}
                  >
                    <sphereGeometry args={[0.002, 30, 30]} />
                    <meshBasicMaterial color="red" />
                  </mesh>
                  <mesh
                    position={[b.center.x, b.center.y, b.center.z]}
                    scale={[b.size.x, b.size.y, b.size.z]}
                    userData={{ isHelper: true }}
                  >
                    <boxGeometry args={[1, 1, 1]} />
                    <meshBasicMaterial color="yellow" wireframe />
                  </mesh>

                  {showCanvasSliders && (
                    <Html
                      position={[
                        b.center.x,
                        b.center.y - b.size.y / 2 - 0.1,
                        b.center.z,
                      ]}
                      center
                    >
                      <div
                        style={{
                          background: "rgba(255,255,255,0.8)",
                          padding: "2px 4px",
                          borderRadius: 3,
                          fontSize: 12,
                        }}
                      >
                        <label># Hulls (non-selected) </label>
                        <SliderWithButtons
                          testId={`canvas-hull-${i}-coarseness`}
                          value={b.coarseness}
                          min={0}
                          max={300}
                          step={1}
                          onChange={(v) => updateBoxCoarsenessValue(i, v)}
                        />
                        <span style={{ marginLeft: 4 }}>
                          {b.coarseness.toFixed(2)}
                        </span>
                      </div>
                    </Html>
                  )}
                </React.Fragment>
              );
            })}
          <FaceSnapControls
            sceneRef={sceneRef}
            activeFace={activeFace}
            setActiveFace={setActiveFace}
            selectedBox={
              selectedBoxIndex !== null ? boundingBoxes[selectedBoxIndex] : null
            }
            updateBox={(updated) =>
              setBoundingBoxes((bs) =>
                bs.map((b, i) => (i === selectedBoxIndex ? updated : b)),
              )
            }
          />
          {unionGeom && (
            <mesh
              geometry={unionGeom}
              renderOrder={999}
              raycast={() => []}
              pointerEvents="none"
            >
              <meshBasicMaterial
                color="orange"
                transparent
                opacity={0.6}
                side={THREE.DoubleSide}
                depthTest={false}
              />
            </mesh>
          )}
        </Suspense>
        <OrbitControls
          ref={orbitRef}
          makeDefault
          target={[0.0416, 0.2287, -0.1461]}
          enableDamping={
            typeof window !== "undefined" &&
            !window.location.search.includes("e2e")
          }
        />
        <Stats />
      </Canvas>
      {showErrorForward && (
        <div
          style={{
            position: "absolute",
            bottom: "70%",
            right: "20%",
            zIndex: 1,
          }}
        >
          <ColorBar
            min={errorMin}
            max={errorMax}
            colors={["white", "#ff0000"]}
          />
        </div>
      )}
      {showErrorReverse && (
        <div
          style={{
            position: "absolute",
            bottom: "70%",
            right: "20%",
            zIndex: 1,
          }}
        >
          <ColorBar
            min={errorMin}
            max={errorMax}
            colors={["white", "#ff0000"]}
          />
        </div>
      )}
      {/* File input (top-left) */}
      <FileLoader onFileSelect={handleFileInput} />

      {/* bottom-left menu */}
      <BottomLeftMenu
        setUnionGeom={setUnionGeom}
        setSumVolume={setSumVolume}
        setUnionVolume={setUnionVolume}
        setCost={setCost}
        allHulls={allHulls}
        setAllHulls={setAllHulls}
        hiddenHulls={hiddenHulls}
        setHiddenHulls={setHiddenHulls}
        onMergeHulls={handleMergeHulls}
        mergeCost={cost}
        sumVolume={sumVolume}
        unionVolume={unionVolume}
        showCanvasSliders={showCanvasSliders}
        setShowCanvasSliders={setShowCanvasSliders}
        errorMin={errorMin}
        errorMax={errorMax}
        setErrorMin={setErrorMin}
        setErrorMax={setErrorMax}
        defaultSize={defaultSize}
        updateDefaultSizeValue={updateDefaultSizeValue}
        defaultCoarseness={defaultCoarseness}
        updateDefaultCoarsenessValue={updateDefaultCoarsenessValue}
        boundingBoxes={boundingBoxes}
        updateBoxSize={updateBoxSize}
        updateBoxPosition={updateBoxPosition}
        updateBoxCoarsenessValue={updateBoxCoarsenessValue}
        handleDeleteBox={handleDeleteBox}
        handleClear={handleClear}
        selectedBoxIndex={selectedBoxIndex}
        setSelectedBoxIndex={setSelectedBoxIndex}
        meshDimensions={meshDimensions}
        selectedHulls={selectedHulls}
        onDeleteHull={deleteHull}
        activeHideTarget={activeHideTarget}
      />
    </>
  );
}
