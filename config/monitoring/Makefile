.PHONY: up down gen-prom logs status clean compose-cmd

# Detect compose command (plugin or legacy)
COMPOSE:=$(shell sh -c 'docker compose version >/dev/null 2>&1 && echo "docker compose" || (command -v docker-compose >/dev/null 2>&1 && echo "docker-compose" || echo "" )')

compose-cmd:
	@if [ -z "$(COMPOSE)" ]; then \
		echo "No docker compose found. Install docker-compose-plugin or docker-compose."; \
		exit 1; \
	fi; \
	$(COMPOSE) version || true

# Start Prometheus and Grafana services
up: compose-cmd
	$(COMPOSE) -f compose/docker-compose.yml up -d

# Stop Prometheus and Grafana services
down: compose-cmd
	$(COMPOSE) -f compose/docker-compose.yml down

# Generate Prometheus config from inventory.yml
gen-prom:
	cd ../../ && python3 src/tatbot/utils/gen_prom_config.py

# Show logs from running services
logs: compose-cmd
	$(COMPOSE) -f compose/docker-compose.yml logs -f

# Show status of services
status: compose-cmd
	$(COMPOSE) -f compose/docker-compose.yml ps

# Clean up volumes and containers
clean: compose-cmd
	$(COMPOSE) -f compose/docker-compose.yml down -v
	docker system prune -f

# Restart services
restart: down up

# Show URLs
urls:
	@echo "Prometheus targets: http://localhost:9090/targets"
	@echo "Grafana dashboard: http://localhost:3000/d/tatbot-compute/tatbot-compute"
	@echo "Kiosk URL: http://eek:3000/d/tatbot-compute/tatbot-compute?kiosk=tv&refresh=5s"
