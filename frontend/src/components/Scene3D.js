"use client";
import React, { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Points, PointMaterial } from "@react-three/drei";
import * as THREE from "three";

function ParticleTunnel({ scrollY }) {
  const pointsRef = useRef();
  
  // Generate particles in a cylinder/tunnel shape
  const [positions, speeds] = useMemo(() => {
    const count = 5000;
    const pos = new Float32Array(count * 3);
    const spd = new Float32Array(count);
    
    for (let i = 0; i < count; i++) {
        // Radius between 2 and 10
        const r = 2 + Math.random() * 8;
        const theta = Math.random() * Math.PI * 2;
        // Z between -100 and 100
        const z = (Math.random() - 0.5) * 200;
        
        pos[i * 3] = r * Math.cos(theta);
        pos[i * 3 + 1] = r * Math.sin(theta);
        pos[i * 3 + 2] = z;
        
        spd[i] = 0.5 + Math.random() * 2;
    }
    return [pos, spd];
  }, []);

  useFrame((state, delta) => {
    if (!pointsRef.current) return;
    
    // Move particles towards the camera (positive Z)
    // increase speed based on scroll
    const scrollMultiplier = 1 + scrollY * 0.05;
    
    const array = pointsRef.current.geometry.attributes.position.array;
    for (let i = 0; i < 5000; i++) {
      array[i * 3 + 2] += speeds[i] * delta * 10 * scrollMultiplier;
      
      // Reset particle if it goes past the camera
      if (array[i * 3 + 2] > 20) {
        array[i * 3 + 2] = -180;
      }
    }
    pointsRef.current.geometry.attributes.position.needsUpdate = true;
    
    // Tilt the whole tunnel slightly based on mouse
    pointsRef.current.rotation.z += delta * 0.1;
  });

  return (
    <Points ref={pointsRef} positions={positions} stride={3} frustumCulled={false}>
      <PointMaterial
        transparent
        color="#00d4ff"
        size={0.05}
        sizeAttenuation={true}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </Points>
  );
}

export default function Scene3D({ scrollY = 0 }) {
  return (
    <div className="blackhole-container">
      <Canvas camera={{ position: [0, 0, 10], fov: 75 }}>
        <color attach="background" args={["#000"]} />
        <fog attach="fog" args={["#000", 20, 100]} />
        <ambientLight intensity={0.5} />
        <ParticleTunnel scrollY={scrollY} />
      </Canvas>
    </div>
  );
}
