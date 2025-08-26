.PHONY: up down gen-prom logs status clean

# Start Prometheus and Grafana services
up:
	docker compose -f compose/docker-compose.yml up -d

# Stop Prometheus and Grafana services
down:
	docker compose -f compose/docker-compose.yml down

# Generate Prometheus config from inventory.yml
gen-prom:
	cd ../../ && python3 scripts/gen_prom_config.py

# Show logs from running services
logs:
	docker compose -f compose/docker-compose.yml logs -f

# Show status of services
status:
	docker compose -f compose/docker-compose.yml ps

# Clean up volumes and containers
clean:
	docker compose -f compose/docker-compose.yml down -v
	docker system prune -f

# Restart services
restart: down up

# Show URLs
urls:
	@echo "Prometheus targets: http://localhost:9090/targets"
	@echo "Grafana dashboard: http://localhost:3000/d/fleet-overview/fleet-overview"
	@echo "Kiosk URL: http://eek:3000/d/fleet-overview/fleet-overview?kiosk=tv&refresh=5s"