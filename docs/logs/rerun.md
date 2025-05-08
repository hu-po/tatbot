## Distributed Rerun Setup Best Practices

https://x.com/i/grok/share/ADQAv7Dp75wSr15xoWqKDixL0
https://chatgpt.com/share/681ce51b-dfcc-8009-8e23-d5ef69f61509

1. **Use one Rerun Viewer as a hub**: 
   - Run a single viewer (with gRPC server) on a capable machine
   - Have all remote processes connect to it for live streaming
   - This centralizes the Chunk Store and visualization, avoiding duplication
   - The viewer acts as the server, receiving data from multiple clients and rendering visualizations
   - Can be started standalone by typing `rerun` in the terminal, listening on default port 9876
   - Use `rr.serve_web_viewer()` to host a web viewer over HTTP that clients can access through a browser
   - For custom server configurations, create a dedicated server script that initializes Rerun with appropriate parameters

2. **Initialize with a shared recording ID**: 
   - Ensure all processes that form one logical session use the same recording_id
   - This helps the viewer know to treat them as one dataset
   - Set a shared recording ID across all clients to ensure data is properly aggregated in the viewer

3. **Prefer connect_grpc for distributed setup**: 
   - On each data source, call `rr.connect_grpc("rerun+http://<viewer-host>:9876")` (or appropriate address) after `rr.init()`
   - This immediately streams logs to the remote viewer
   - Do not use `spawn=True` on headless devices (it would attempt to open a viewer there)
   - Consider using `rr.serve_grpc()` on the server to create a dedicated gRPC endpoint for clients
   - Each computer uses the Rerun SDK to connect to the viewer's address (e.g., `rr.connect("http://<visualization-computer-ip>:9876")` in Python)

4. **Minimize device overhead**: 
   - Log only necessary data at appropriate rates
   - The Rerun SDK is lightweight, but sending huge data at high frequency can strain any system
   - Use Rerun's efficient logging of scalar/tensor data for things like joint angles
   - Only send high-bandwidth data (images, point clouds) when needed or in compressed form if possible
   - Each data-generating computer logs data (e.g., camera feeds, robot arm positions) and sends it to the viewer

5. **Monitor network and memory**: 
   - On the central viewer, keep an eye on memory usage – a long run with unlimited data will eventually consume a lot of RAM
   - You can periodically clear old data if needed (Rerun supports "tombstone" deletion of older data in the store) or simply restart the viewer between runs
   - Network-wise, a wired connection is recommended for multiple video streams

6. **Persistence if needed**: 
   - Decide ahead if you need to persist data
   - For quick debugging, transient mode is fine
   - If you need to analyze later, plan to use `rr.save()` on each process (writing multiple files that you can later merge) or record via an alternate mechanism
   - Keep Rerun versions in sync if using .rrd files, due to format instability

7. **Launching with Docker**: 
   - No special Docker configuration is required beyond making sure the container can reach the viewer's IP
   - Outbound connection to the viewer's port 9876 is all that's needed
   - If using Docker Compose, you might parameterize the viewer address
   - Ensure time synchronization across machines if cross-referencing timestamps (use NTP or share a common clock source if precise alignment is important for sensor data)
   - Expose ports in Docker to ensure clients can reach the viewer
