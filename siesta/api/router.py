from siesta.core.app import Siesta
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


"""
API Router for Siesta Framework.
"""


def startup(siestaInstance: Siesta) -> None:
    """Startup hook to register routes from discovered modules."""
    config = siestaInstance.config
    app = FastAPI(title=config.get("app_name", "Siesta Framework API"))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    routes = siestaInstance.get_registered_routes()

    for module, registered_routes in routes.items():
        if not registered_routes:
            continue
        module_router = APIRouter(tags=[module])
        for endpoint, route_def in registered_routes.items():
            options = {}
            if len(route_def) == 3:
                method, func, options = route_def
            else:
                method, func = route_def

            route_path = f"/{module}/{endpoint}"

            match method:
                case "GET":
                    module_router.get(route_path, **options)(func)
                case "POST":
                    module_router.post(route_path, **options)(func)
                case "PUT":
                    module_router.put(route_path, **options)(func)
                case "DELETE":
                    module_router.delete(route_path, **options)(func)
                case _:
                    raise ValueError(f"Unsupported API method: {method}")

        app.include_router(module_router)
    
    
    health_router = APIRouter(tags=["health"])
    @health_router.get("/health")
    def health_check():
        return {"status": "ok", "message": "Siesta API is healthy."}
    app.include_router(health_router)

    run_config = config.get("api", {"host": "localhost", "port": 8000})
    uvicorn.run(app, host=run_config["host"], port=run_config["port"])
