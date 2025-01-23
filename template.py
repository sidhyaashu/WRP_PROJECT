import os

# Define the folder structure
folders = [
    "app",
    "app/api",
    "app/api/routes",
    "app/api/dependencies",
    "app/core",
    "app/db",
    "app/services",
    "app/schemas",
    "app/utils",
    "tests"
]

# Define initial files with sample content
files = {
    "app/__init__.py": "",
    "app/main.py": """from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI project!"}
""",
    "app/api/__init__.py": "",
    "app/api/routes/__init__.py": "",
    "app/api/routes/example.py": """from fastapi import APIRouter

router = APIRouter()

@router.get("/example")
def read_example():
    return {"message": "This is an example endpoint"}
""",
    "app/api/dependencies/__init__.py": "",
    "app/api/dependencies/auth.py": """def get_current_user():
    return {"user_id": "test_user"}
""",
    "app/core/__init__.py": "",
    "app/core/config.py": """from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "FastAPI Project"
    database_url: str = "sqlite:///./test.db"

    class Config:
        env_file = ".env"

settings = Settings()
""",
    "app/core/security.py": """def verify_token(token: str):
    return token == "valid_token"
""",
    "app/db/__init__.py": "",
    "app/db/models.py": """from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ExampleModel(Base):
    __tablename__ = "examples"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
""",
    "app/db/session.py": """from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
""",
    "app/services/__init__.py": "",
    "app/services/example_service.py": """def get_example_data():
    return {"data": "example"}
""",
    "app/schemas/__init__.py": "",
    "app/schemas/example_schema.py": """from pydantic import BaseModel

class ExampleSchema(BaseModel):
    id: int
    name: str
""",
    "app/utils/__init__.py": "",
    "app/utils/file_utils.py": """def read_file(file_path: str):
    with open(file_path, 'r') as file:
        return file.read()
""",
    "tests/__init__.py": "",
    "tests/test_example.py": """def test_example():
    assert True
""",
    ".env": "APP_NAME=FastAPI Project\nDATABASE_URL=sqlite:///./test.db",
    "requirements.txt": "fastapi\nuvicorn\nsqlalchemy\npydantic\npytest",
    "README.md": "# FastAPI Project\n\nA boilerplate for building REST APIs with FastAPI."
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for file_path, content in files.items():
    with open(file_path, "w") as f:
        f.write(content)

print("Project structure created successfully!")
