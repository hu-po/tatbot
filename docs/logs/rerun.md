

https://x.com/i/grok/share/ADQAv7Dp75wSr15xoWqKDixL0

- expose ports in Docker to ensure clients can reach the viewer.
- Each computer, containerized with Docker, uses the Rerun SDK to connect to the viewer's address. For example, in Python, use rr.connect("http://<visualization-computer-ip>:9876") to establish the connection, then log data using the RecordingStream API.
- Client Role: Each data-generating computer, running in Docker containers, logs data (e.g., camera feeds, robot arm positions) and sends it to the viewer. Functions like rr.connect in Python facilitate this connection to a remote viewer.
- Server Role: The Rerun Viewer, typically run on the visualization computer, acts as the server, receiving data from multiple clients and rendering visualizations. It can be started standalone by typing rerun in the terminal, listening on a default port (e.g., 9876).

https://chatgpt.com/share/681ce51b-dfcc-8009-8e23-d5ef69f61509

For a distributed robotic system, Rerun’s recommended pattern is to use a centralized logging server – i.e. one Rerun viewer instance – and have all remote sources stream directly to it. In this pattern, each machine (Raspberry Pis, NVIDIA Orin, System76 Meerkat, etc.) runs its sensor/robot code instrumented with Rerun SDK logging. Instead of attempting to visualize locally, they all act as clients that directly stream their data to the central viewer over the LAN. This centralized viewer aggregates all streams in real-time, so you get a unified visualization of the entire system’s state.


