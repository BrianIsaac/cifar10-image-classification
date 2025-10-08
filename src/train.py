"""
This script is for training a model on the CIFAR10 dataset.
"""
# Standard library
import json
import logging
import math
import os
import gc

# Third-party libraries
import albumentations as A
import hydra
from hydra.utils import instantiate
import mlflow
from omegaconf import OmegaConf
import torch
import torch.optim.lr_scheduler as lr_scheduler

# Local application imports
import general_utils
from utils import ImageLoader, CIFARClassifier, Trainer

# pylint: disable = no-value-for-parameter
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(args):
    """This is the main function for training the model.

    Parameters
    ----------
    args : omegaconf.DictConfig
        An omegaconf.DictConfig object containing arguments for the main function.
    """

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    general_utils.setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        ),
        log_dir=args.get("log_dir", None),
    )

    ## Initialize MLflow (Use the appropriate function from general_utils.py)
    # mlflow_init_status, mlflow_run = general_utils.<method>
    mlflow_init_status, mlflow_run, step_offset = general_utils.mlflow_init(
        tracking_uri=args.mlflow_tracking_uri, 
        exp_name=args.mlflow_exp_name, 
        run_name=args.mlflow_run_name, 
        setup_mlflow=True, 
        autolog=True,
        resume=args.resume
    )

    ## Log hyperparameters used to train the model
    # general_utils.<method>
    general_utils.mlflow_log(
        mlflow_init_status,
        log_function="log_params",
        params = {
            "lr": args.lr,
            "epochs": args.epochs,
            "train_bs": args.train_bs,
            "seed": args.seed,
            "model_architecture": args.cv_architecture,
            "image_size": args.image_size,
            "freeze_backbone":args.freeze_backbone,
            "model_checkpoint_dir_path": args.model_checkpoint_dir_path,
            "optimiser": args.optimiser
        }
    )

    torch.manual_seed(args["seed"])

    # specify transformations on training set
    train_transform = A.Compose([instantiate(t) for t in args.transforms.train])

    loader = ImageLoader(
        dataset_name="cifar10",
        data_dir="./data",
        train_transform=train_transform,
        val_transform=train_transform,
        batch_size=args.train_bs,
        image_size=args.image_size
    )

    # initialise model
    classifier = CIFARClassifier(base_model_name=args.cv_architecture,
                                 freeze_backbone=args.freeze_backbone, 
                                 device=args.device)
    
    # compile model
    classifier = torch.compile(classifier)

    # enable matmul
    torch.set_float32_matmul_precision('high')

    # return train_loader and val_loader
    train_loader, val_loader = loader.get_loaders()

    # logging batch size
    logger.info(f"Train loader: {len(train_loader)} batches | Val loader: {len(val_loader)} batches")

    # log for device used
    logger.info(f"Using device: {args.device}, for training")

    if args.optimiser == "Adam":
        optimiser = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    elif args.optimiser == "AdamW":
        optimiser = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Setup warm-up and cosine decay scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps as warm-up

    scheduler = lr_scheduler.SequentialLR(
        optimiser,
        schedulers=[
            lr_scheduler.LinearLR(optimiser, start_factor=0.1, total_iters=warmup_steps),
            lr_scheduler.CosineAnnealingLR(optimiser, T_max=total_steps - warmup_steps)
        ],
        milestones=[warmup_steps]
    )

    # Default start epoch
    start_epoch = 1

    # Check for resume flag
    model_checkpoint_path = os.path.join(
        args["model_checkpoint_dir_path"], "Model", "cifar10.pth"
    )

    if args.resume and os.path.exists(model_checkpoint_path):
        logger.info("Resuming training from checkpoint...")
        checkpoint = torch.load(model_checkpoint_path)

        classifier.load_state_dict(checkpoint["model_state_dict"])
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resumed from epoch {checkpoint['epoch']}. Continuing from epoch {start_epoch}.")

    else:
        logger.info("No checkpoint found. Starting training from scratch.")

    # initialise trainer
    trainer = Trainer(
        model=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        optimizer=optimiser
    )

    # set checkpoint
    model_checkpoint = math.ceil(args.epochs * 0.01)

    # Run training loop and preserve MLflow logic
    for epoch in range(start_epoch, args.epochs+1):
        logger.info(f"Epoch {epoch}/{args['epochs']} starting...")

        # Train and evaluate for one epoch
        train_loss, train_acc = trainer.train_one_epoch(use_tqdm=True, scheduler=scheduler)
        val_loss, val_acc = trainer.val_losses[-1], trainer.val_accuracies[-1]

        # Save checkpoint only every 10 epochs
        if epoch % model_checkpoint == 0 and epoch != 0:
            logger.info("Saving checkpoint at epoch %s.", epoch)

            # Log metric to MLFlow 
            general_utils.mlflow_log(
                mlflow_init_status,
                log_function="log_metric",
                key="train_loss",
                value=train_loss,
                step=epoch + step_offset,
            )
            general_utils.mlflow_log(
                mlflow_init_status,
                log_function="log_metric",
                key="train_accuracy",
                value=train_acc,
                step=epoch + step_offset,
            )
            general_utils.mlflow_log(
                mlflow_init_status,
                log_function="log_metric",
                key="val_loss",
                value=val_loss,
                step=epoch + step_offset,
            )
            general_utils.mlflow_log(
                mlflow_init_status,
                log_function="log_metric",
                key="val_accuracy",
                value=val_acc,
                step=epoch + step_offset,
            )

            artifact_subdir = "Model"
            model_checkpoint_path = os.path.join(
                args["model_checkpoint_dir_path"], artifact_subdir, "cifar10.pth"
            )
            os.makedirs(os.path.dirname(model_checkpoint_path), exist_ok=True)

            torch.save(
                {
                    "model_state_dict": classifier.state_dict(),
                    "epoch": epoch,
                    "optimiser_state_dict": trainer.optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                model_checkpoint_path,
            )

            ## Use MLflow to log artifact (model checkpoint)
            # general_utils.<method>
            general_utils.mlflow_log(
                mlflow_init_status, 
                log_function="log_artifact",
                local_path=model_checkpoint_path, 
                artifact_path=artifact_subdir
            )

    ## Use MLflow to log artifact (model config in json)
    # general_utils.<method>
    config_path = os.path.join(args.model_checkpoint_dir_path, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(
            OmegaConf.to_container(args, resolve=True),
            f,
            indent=2
        )

    general_utils.mlflow_log(
        mlflow_init_status,
        log_function="log_artifact",
        local_path=config_path
    )

    ## Use MLflow to log artifacts (entire `logs`` directory)
    # general_utils.<method>
    general_utils.mlflow_log(
        mlflow_init_status,
        log_function="log_artifacts",
        local_dir=args.log_dir
    )

    # Perform final evaluation and save classification report
    report_dict = trainer.final_evaluation()

    # Save classification report as JSON
    report_path = os.path.join(args.model_checkpoint_dir_path, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    # Log report as MLflow artifact
    general_utils.mlflow_log(
        mlflow_init_status,
        log_function="log_artifact",
        local_path=report_path
    )

    # Save misclassified images and log to MLflow
    misclassified_output_dir = os.path.join(args.model_checkpoint_dir_path, "misclassified_samples")
    trainer.save_misclassified(
        output_dir=misclassified_output_dir,
        class_names=loader.get_class_names(),
        max_images=10
    )

    # Log folder and metadata file as artifacts to MLflow
    general_utils.mlflow_log(
        mlflow_init_status,
        log_function="log_artifacts",
        local_dir=misclassified_output_dir
    )

    ## Use MLflow to log model for model registry, using pytorch specific methods
    # general_utils.<method>

    general_utils.mlflow_pytorch_call(
        mlflow_init_status,
        pytorch_function="log_model",
        pytorch_model=classifier, 
        artifact_path='Model',
    )

    if mlflow_init_status:
        ## Get artifact link
        artifact_uri = mlflow.get_artifact_uri()
        logger.info("Artifact URI: %s", artifact_uri)
        general_utils.mlflow_log(
            mlflow_init_status, "log_params", params={"artifact_uri": artifact_uri}
        )
        logger.info(
            "Model training with MLflow run ID %s has completed.",
            mlflow_run.info.run_id,
        )
        mlflow.end_run()
    else:
        logger.info("Model training has completed.")

    logger.info(f"GPU memory allocated before cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    logger.info(f"GPU memory reserved before cleanup: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # Delete trainer and classifier explicitly
    del trainer
    del classifier

    # Empty CUDA cache and run garbage collection
    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"GPU memory allocated after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    logger.info(f"GPU memory reserved after cleanup: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    logger.info("GPU memory cleanup completed.")

    # Outputs for conf/train_model.yaml for hydra.sweeper.direction
    return val_loss, val_acc


if __name__ == "__main__":
    main()
