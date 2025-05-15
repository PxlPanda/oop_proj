.PHONY: build-frontend run-backend run

build-frontend:
	cd frontend && npm run build

run-backend:
	cd backend && . ../venv/bin/activate && python manage.py runserver

run: build-frontend run-backend
