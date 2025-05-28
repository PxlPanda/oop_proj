.PHONY: build-frontend run-backend run

build-frontend:
	cd frontend && npm run build

run-backend:
	cd backend && . ../venv/bin/activate && python manage.py runserver

run: build-frontend run-backend

run-backend-win:
	cd backend && ..\venv\Scripts\python.exe manage.py runserver

#Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
