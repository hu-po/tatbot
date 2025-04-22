import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { USDZLoader } from 'three/addons/loaders/USDZLoader.js';

let scene, camera, renderer, orbitControls, transformControls;
let INTERACTIVE_OBJECTS = []; // Store objects we can interact with
let selectedObject = null;

const container = document.getElementById('viewer-container');

function init() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xcccccc); // Light grey background
    scene.fog = new THREE.FogExp2(0xcccccc, 0.002);

    // Camera
    camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(5, 5, 5); // Adjust starting position

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Controls - Orbit
    orbitControls = new OrbitControls(camera, renderer.domElement);
    orbitControls.enableDamping = true; // an animation loop is required when either damping or auto-rotation are enabled
    orbitControls.dampingFactor = 0.05;
    orbitControls.screenSpacePanning = false;
    orbitControls.minDistance = 1;
    orbitControls.maxDistance = 500;
    orbitControls.maxPolarAngle = Math.PI / 2; // Prevent looking from below ground
    orbitControls.target.set(0, 1, 0); // Point camera towards this point
    orbitControls.update();

    // Controls - Transform (Gizmo)
    transformControls = new TransformControls(camera, renderer.domElement);
    transformControls.addEventListener('dragging-changed', function (event) {
         // Disable OrbitControls while dragging an object
         orbitControls.enabled = !event.value;
    });
    transformControls.addEventListener('mouseUp', function () {
        // Event listener when manipulation ends
        if (selectedObject) {
            console.log('Transform finished for:', selectedObject.name || selectedObject.uuid);
            sendTransformUpdate(selectedObject);
        }
    });
    // Optional: Add finer-grained control listeners like 'objectChange' if needed during drag
    scene.add(transformControls);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x606060); // soft white light
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 0).normalize();
    scene.add(directionalLight);

    // Grid Helper
    const gridHelper = new THREE.GridHelper(10, 10); // Size 10x10 grid
    scene.add(gridHelper);

    // Load the USD model
    loadScene();

    // Event Listeners
    window.addEventListener('resize', onWindowResize);
    renderer.domElement.addEventListener('click', onClick); // Use renderer canvas for clicks

    // Start animation loop
    animate();
}

function loadScene() {
    console.log("Fetching scene data...");
    // Get the filename from the environment variable
    let filename = window.USD_FILE || 'stencil.usdz';
    // Remove 'output/' prefix if present
    if (filename.startsWith('output/')) {
        filename = filename.substring(7);
    }
    
    fetch(`/api/scene_data?file=${encodeURIComponent(filename)}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                console.error("Error loading scene data:", data.error);
                alert("Error loading scene data: " + data.error);
                return;
            }
            if (data.usd_url) {
                console.log("Loading USD:", data.usd_url);
                const loader = new USDZLoader();
                
                // Add warning handler
                loader.onWarning = (warning) => {
                    console.warn('USDZLoader warning:', warning);
                    // Don't show alert for warnings as they might be expected
                };
                
                loader.load(
                    data.usd_url,
                    (group) => {
                        // USDZLoader often returns a group, find the main object(s)
                        const box = new THREE.Box3().setFromObject(group);
                        const center = box.getCenter(new THREE.Vector3());
                        group.position.sub(center); // Center the model
                        scene.add(group);

                        // Check if we actually got any meshes
                        let hasMeshes = false;
                        group.traverse((child) => {
                            if (child.isMesh) {
                                hasMeshes = true;
                                console.log("Found interactive mesh:", child.name || child.uuid);
                                child.userData.originalMaterial = child.material;
                                INTERACTIVE_OBJECTS.push(child);
                            }
                        });

                        if (!hasMeshes) {
                            console.warn("No meshes found in the loaded model");
                        }

                        console.log('USD loaded successfully.');
                        orbitControls.target.copy(group.position);
                        orbitControls.update();
                    },
                    (xhr) => {
                        console.log(`USD loading: ${(xhr.loaded / xhr.total * 100).toFixed(2)}%`);
                    },
                    (error) => {
                        console.error('Error loading USD:', error);
                        alert("Failed to load USD model. Check console for details.");
                    }
                );
            } else {
                console.error("No usd_url found in response.");
                alert("Backend did not provide a scene URL.");
            }
        })
        .catch(error => {
            console.error('Error fetching scene data:', error);
            alert("Could not fetch scene data from backend. Is it running?");
        });
}

function onClick(event) {
    // Only raycast if not dragging the transform control
    if (transformControls.dragging) return;

    const mouse = new THREE.Vector2();
    // Calculate mouse position in normalized device coordinates (-1 to +1) for component
    mouse.x = (event.clientX / container.clientWidth) * 2 - 1;
    mouse.y = -(event.clientY / container.clientHeight) * 2 + 1;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, camera);

    const intersects = raycaster.intersectObjects(INTERACTIVE_OBJECTS, false); // Check only interactive objects

    if (intersects.length > 0) {
        const object = intersects[0].object
        // Check if the object is already selected
        if (selectedObject !== object) {
            selectObject(object);
        }
    } else {
        // Clicked on empty space, deselect
        deselectObject();
    }
}

function selectObject(object) {
    if (selectedObject) {
         // Restore previous object's appearance if needed (optional)
         selectedObject.material = selectedObject.userData.originalMaterial || selectedObject.material;
    }

    selectedObject = object;
    // Highlight selected object (optional) - e.g., change material color or use an outline effect
    if (selectedObject.material) {
        selectedObject.material = selectedObject.material.clone(); // Clone to avoid modifying original
        selectedObject.material.emissive = new THREE.Color(0xaaaa00); // Yellow emissive highlight
    }

    transformControls.attach(selectedObject); // Attach gizmo
    console.log("Selected:", selectedObject.name || selectedObject.uuid);
}

function deselectObject() {
    if (selectedObject) {
         // Restore appearance
         if (selectedObject.material && selectedObject.userData.originalMaterial) {
            selectedObject.material = selectedObject.userData.originalMaterial;
         } else if (selectedObject.material) {
              // Simple fallback if original wasn't stored properly
              selectedObject.material.emissive = new THREE.Color(0x000000);
         }
    }
    selectedObject = null;
    transformControls.detach(); // Detach gizmo
    console.log("Deselected");
}

function sendTransformUpdate(object) {
    if (!object) return;

    const updateData = {
         // Use UUID for uniqueness, fallback to name if needed (ensure names are unique!)
         objectId: object.uuid,
         objectName: object.name || null, // Include name for reference
         position: object.position.toArray(),
         // Use quaternion for rotation - more robust than Euler angles
         quaternion: object.quaternion.toArray(),
         scale: object.scale.toArray()
    };

    console.log("Sending update:", updateData);

    fetch('/api/update_transform', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(updateData),
    })
    .then(response => response.json())
    .then(data => {
        console.log('Backend response:', data);
        if (data.status !== 'success') {
             console.error("Error updating transform on backend:", data.error);
             // Optional: Add user feedback about the save failure
        }
    })
    .catch((error) => {
        console.error('Error sending update to backend:', error);
        // Optional: Add user feedback about the connection error
    });
}

function onWindowResize() {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
    requestAnimationFrame(animate);
    orbitControls.update(); // only required if controls.enableDamping = true, or if controls.autoRotate = true
    // transformControls updates internally? Check docs if needed.
    render();
}

function render() {
    renderer.render(scene, camera);
}

// Start the application
init(); 