"""
Evaluation Service API - Serves evaluation results and metrics

This service provides REST API endpoints to access evaluation results
from the predictions mode evaluation runs.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Translation Evaluation API",
    version="1.0.0",
    description="API for accessing evaluation results and metrics"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Results directory
RESULTS_DIR = Path(__file__).parent / 'results'


class ExecutionSummary(BaseModel):
    """Summary of an evaluation execution"""
    execution_id: str
    timestamp: str
    nmt_model: str
    tts_model: str
    metrics: List[str]
    languages: Dict[str, Dict]
    total_samples: int
    total_valid_samples: int
    overall_summary: Optional[Dict] = None


class LanguageResults(BaseModel):
    """Results for a specific language"""
    language: str
    summary: Dict
    visualizations: List[str]


class ExecutionVisualizationsResponse(BaseModel):
    """List of execution-level visualizations"""
    execution_id: str
    visualizations: List[str]


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "evaluation_api",
        "results_directory": str(RESULTS_DIR),
        "results_exists": RESULTS_DIR.exists()
    }


@app.get("/executions", response_model=List[ExecutionSummary])
async def list_executions():
    """
    List all evaluation executions.

    Returns executions in reverse chronological order (newest first).
    Only returns execution-based results (eval_YYYYMMDD_HHMMSS format).
    """
    try:
        if not RESULTS_DIR.exists():
            return []

        executions = []

        # Iterate through results directory
        for exec_dir in sorted(RESULTS_DIR.iterdir(), reverse=True):
            # Only process execution-based directories (eval_*)
            if not exec_dir.is_dir() or not exec_dir.name.startswith('eval_'):
                continue

            manifest_path = exec_dir / 'manifest.json'
            if not manifest_path.exists():
                logger.warning(f"No manifest found for {exec_dir.name}")
                continue

            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)

                executions.append(ExecutionSummary(**manifest))
            except Exception as e:
                logger.warning(f"Could not load execution {exec_dir.name}: {e}")

        return executions

    except Exception as e:
        logger.error(f"Error listing executions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not list executions: {str(e)}"
        )


@app.get("/executions/{execution_id}", response_model=ExecutionSummary)
async def get_execution(execution_id: str):
    """
    Get detailed information about a specific execution.

    Returns the manifest with overall summary and all language results.
    """
    try:
        exec_dir = RESULTS_DIR / execution_id

        if not exec_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found"
            )

        manifest_path = exec_dir / 'manifest.json'
        if not manifest_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Manifest not found for execution '{execution_id}'"
            )

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        return ExecutionSummary(**manifest)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution {execution_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not get execution: {str(e)}"
        )


@app.get("/executions/{execution_id}/visualizations", response_model=ExecutionVisualizationsResponse)
async def get_execution_visualizations(execution_id: str):
    """
    Get list of execution-level visualizations.
    """
    try:
        exec_dir = RESULTS_DIR / execution_id
        viz_dir = exec_dir / 'visualizations'

        if not exec_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found"
            )

        visualizations = []
        if viz_dir.exists():
            visualizations = [f.name for f in viz_dir.glob('*.png')]

        return ExecutionVisualizationsResponse(
            execution_id=execution_id,
            visualizations=visualizations
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting visualizations for {execution_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not get visualizations: {str(e)}"
        )


@app.get("/executions/{execution_id}/languages/{language}", response_model=LanguageResults)
async def get_language_results(execution_id: str, language: str):
    """
    Get results for a specific language within an execution.

    Returns summary, metrics, and list of available visualizations.
    """
    try:
        exec_dir = RESULTS_DIR / execution_id
        lang_dir = exec_dir / language

        if not lang_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Language '{language}' not found in execution '{execution_id}'"
            )

        # Load summary
        summary_path = lang_dir / 'summary.json'
        if not summary_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Summary not found for {language} in execution '{execution_id}'"
            )

        with open(summary_path, 'r') as f:
            summary = json.load(f)

        # List visualizations
        viz_dir = lang_dir / 'visualizations'
        visualizations = []
        if viz_dir.exists():
            visualizations = [f.name for f in viz_dir.glob('*.png')]

        return LanguageResults(
            language=language,
            summary=summary,
            visualizations=visualizations
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting language results for {language} in {execution_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not get language results: {str(e)}"
        )


@app.get("/executions/{execution_id}/languages/{language}/files/{filename}")
async def get_language_file(execution_id: str, language: str, filename: str):
    """
    Serve files for a specific language (visualizations, CSV, JSON, HTML).

    Security: Validates filename to prevent directory traversal.
    """
    try:
        # Validate filename to prevent directory traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid filename"
            )

        lang_dir = RESULTS_DIR / execution_id / language

        if not lang_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Language '{language}' not found in execution '{execution_id}'"
            )

        # Try visualizations directory first
        viz_file = lang_dir / 'visualizations' / filename
        if viz_file.exists() and viz_file.is_file():
            return FileResponse(viz_file)

        # Try language results directory
        lang_file = lang_dir / filename
        if lang_file.exists() and lang_file.is_file():
            return FileResponse(lang_file)

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{filename}' not found for {language} in execution '{execution_id}'"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving file {filename} for {language} in {execution_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not serve file: {str(e)}"
        )


@app.get("/executions/{execution_id}/files/{filename}")
async def get_execution_file(execution_id: str, filename: str):
    """
    Serve execution-level files (manifest.json, overall_summary.json, visualizations).

    Security: Validates filename to prevent directory traversal.
    """
    try:
        # Validate filename to prevent directory traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid filename"
            )

        exec_dir = RESULTS_DIR / execution_id

        if not exec_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found"
            )

        # Try visualizations directory first (for execution-level charts)
        viz_file = exec_dir / 'visualizations' / filename
        if viz_file.exists() and viz_file.is_file():
            return FileResponse(viz_file)

        # Try execution root directory
        file_path = exec_dir / filename
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{filename}' not found in execution '{execution_id}'"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving file {filename} for execution {execution_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not serve file: {str(e)}"
        )


if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("EVALUATION_API_PORT", 8079))

    logger.info(f"Starting Evaluation API on port {port}...")
    logger.info("=" * 60)
    logger.info(f"Results directory: {RESULTS_DIR}")
    logger.info("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
