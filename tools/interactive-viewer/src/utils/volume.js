// Copyright 2025 Intrinsic Innovation LLC

// ---------------------------------------------------------------------------
// utils/volume.js
// ---------------------------------------------------------------------------
// This module provides utility functions for computing the convex hull volume

import convexHull from "convex-hull";
import * as THREE from "three";

/**
 * Extracts world-space points from a Three.js mesh.
 */
export function pointsFromMesh(mesh) {
  mesh.updateMatrixWorld();
  // clone geometry & bake world transform
  const geo = mesh.geometry.clone().applyMatrix4(mesh.matrixWorld);
  const arr = geo.attributes.position.array;
  const pts = [];
  for (let i = 0; i < arr.length; i += 3) {
    pts.push([arr[i], arr[i + 1], arr[i + 2]]);
  }
  return pts;
}

/**
 * Given an array of [x,y,z] points, compute
 * the volume of their convex hull.
 */
export function computeConvexHullVolumeFromPoints(pts) {
  const faces = convexHull(pts); // returns [[i,j,k],…]
  let vol = 0;
  const v0 = new THREE.Vector3();
  const v1 = new THREE.Vector3();
  const v2 = new THREE.Vector3();

  for (const [i, j, k] of faces) {
    v0.fromArray(pts[i]);
    v1.fromArray(pts[j]);
    v2.fromArray(pts[k]);
    // volume of tetrahedron (0, v0, v1, v2) = dot(v0, cross(v1,v2)) / 6
    vol += v0.dot(v1.clone().cross(v2));
  }
  return Math.abs(vol) / 6;
}

/**
 * Given an array of Three.js Mesh instances,
 * extracts all their points and returns the
 * convex-hull volume of their union.
 */
export function computeMergedHullVolume(meshes) {
  const allPts = meshes.flatMap(pointsFromMesh);
  return computeConvexHullVolumeFromPoints(allPts);
}
