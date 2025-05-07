@echo off
REM Script to run the entire workflow from data extraction to federated learning

REM Set default values
set DATASET_DIR=data\extracted
set PARTITIONED_DIR=data\partitioned
set NUM_ROUNDS=10
set USE_DP=
set ALLOW_NEW_CLASSES=--add_unseen_classes
set NUM_UNSEEN_CLASSES=5
set CLEAN_DOCKER=--clean

REM Parse command line arguments
:args_loop
if "%~1" == "" goto args_done
if "%~1" == "--dataset_dir" (
    set DATASET_DIR=%~2
    shift
    shift
    goto args_loop
)
if "%~1" == "--partitioned_dir" (
    set PARTITIONED_DIR=%~2
    shift
    shift
    goto args_loop
)
if "%~1" == "--num_rounds" (
    set NUM_ROUNDS=%~2
    shift
    shift
    goto args_loop
)
if "%~1" == "--use_dp" (
    set USE_DP=--use_dp
    shift
    goto args_loop
)
if "%~1" == "--no_allow_new_classes" (
    set ALLOW_NEW_CLASSES=
    shift
    goto args_loop
)
if "%~1" == "--num_unseen_classes" (
    set NUM_UNSEEN_CLASSES=%~2
    shift
    shift
    goto args_loop
)
if "%~1" == "--no_clean_docker" (
    set CLEAN_DOCKER=
    shift
    goto args_loop
)
if "%~1" == "--help" (
    echo Usage: %0 [options]
    echo Options:
    echo   --dataset_dir DIR        Path to extracted dataset (default: data\extracted)
    echo   --partitioned_dir DIR    Path for partitioned data (default: data\partitioned)
    echo   --num_rounds N           Number of federated learning rounds (default: 10)
    echo   --use_dp                 Enable differential privacy
    echo   --no_allow_new_classes   Disable handling of new classes by clients
    echo   --num_unseen_classes N   Number of unseen classes per client (default: 5)
    echo   --no_clean_docker        Don't clean Docker environment before starting
    exit /b 0
)

echo Unknown option: %~1
exit /b 1

:args_done

echo =========== Privacy-Preserving Facial Recognition Federated Learning ===========
echo Dataset directory: %DATASET_DIR%
echo Partitioned data directory: %PARTITIONED_DIR%
echo Number of rounds: %NUM_ROUNDS%
echo Use differential privacy: %USE_DP%
echo Allow new classes: %ALLOW_NEW_CLASSES%
echo Number of unseen classes: %NUM_UNSEEN_CLASSES%
echo Clean Docker environment: %CLEAN_DOCKER%
echo ==============================================================================

REM Check if dataset exists
if not exist "%DATASET_DIR%" (
    echo Error: Dataset directory %DATASET_DIR% does not exist.
    echo Please extract the CASIA-WebFace dataset first using:
    echo python run.py extract --rec_file path/to/train.rec --idx_file path/to/train.idx --lst_file path/to/train.lst --output_dir %DATASET_DIR%
    exit /b 1
)

REM Step 1: Train and save initial models
echo.
echo ==== Step 1: Training initial models ====

set CLASS_PARAMS=
if not "%ALLOW_NEW_CLASSES%" == "" (
    set CLASS_PARAMS=--add_unseen_classes --num_unseen_classes %NUM_UNSEEN_CLASSES%
)

set TRAIN_CMD=python train_and_save_models.py --dataset_dir %DATASET_DIR% --output_dir %PARTITIONED_DIR% %USE_DP% %CLASS_PARAMS%
echo Running: %TRAIN_CMD%
%TRAIN_CMD%

if errorlevel 1 (
    echo Error: Initial model training failed.
    exit /b 1
)

REM Step 2: Deploy federated learning in Docker
echo.
echo ==== Step 2: Deploying federated learning in Docker ====

set DEPLOY_CMD=python deploy_federated_learning.py --model_dir models --data_dir %PARTITIONED_DIR% --num_rounds %NUM_ROUNDS% %USE_DP% %ALLOW_NEW_CLASSES% %CLEAN_DOCKER%
echo Running: %DEPLOY_CMD%
%DEPLOY_CMD%

if errorlevel 1 (
    echo Error: Deployment of federated learning failed.
    exit /b 1
)

echo.
echo Federated learning deployment complete!
echo To view logs: docker-compose logs -f
echo To stop containers: docker-compose down 