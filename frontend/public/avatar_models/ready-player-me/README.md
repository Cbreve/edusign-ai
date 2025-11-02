# Ready Player Me Avatar

This directory contains the 3D avatar model created with Ready Player Me for sign language animation.

## File Structure

```
ready-player-me/
├── models/           # 3D model files (.glb, .gltf)
├── textures/         # Texture maps (diffuse, normal, roughness, etc.)
└── animations/       # Sign language animation files (if separate)
```

## Usage in Frontend

### Loading with Three.js

```javascript
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

const loader = new GLTFLoader();
loader.load('/avatar_models/ready-player-me/models/avatar.glb', (gltf) => {
  const avatar = gltf.scene;
  scene.add(avatar);
});
```

### Loading with React Three Fiber

```jsx
import { useGLTF } from '@react-three/drei';

function Avatar() {
  const { scene } = useGLTF('/avatar_models/ready-player-me/models/avatar.glb');
  return <primitive object={scene} />;
}
```

## File Locations

- **Main Model**: `models/avatar.glb`
- **Textures**: `textures/*.jpg` or `textures/*.png`
- **Animations**: `animations/*.glb` (if separate animation files)

## Notes

- GLB format is recommended for web (smaller file size, includes textures)
- Ensure textures are optimized for web (compressed JPEG/PNG)
- Avatar should support sign language gestures/animations

