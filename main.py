import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, pin_memory=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, pin_memory=True, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, pin_memory=True, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    model = BasicConvClassifier(
        train_set.num_classes, train_set.seq_len, train_set.num_channels, train_set.num_subjects
    ).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # ------------------
    #   Start training
    # ------------------  
    accuracy_class = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
    
    accuracy_subject = Accuracy(
        task="multiclass", num_classes=train_set.num_subjects
    ).to(args.device)
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc_class, train_acc_subject, val_loss, val_acc_class, val_acc_subject = [], [], [], [], [], []

        
        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y, subject_idxs = X.to(args.device), y.to(args.device), subject_idxs.to(args.device)

            class_logits, subject_logits = model(X)
            
            loss_class = F.cross_entropy(class_logits, y)
            loss_subject = F.cross_entropy(subject_logits, subject_idxs)
            loss = loss_class + 0.5*loss_subject  # Combined loss
            
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc_class = accuracy_class(class_logits, y)
            acc_subject = accuracy_subject(subject_logits, subject_idxs)
            
            train_acc_class.append(acc_class.item())
            train_acc_subject.append(acc_subject.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y, subject_idxs = X.to(args.device), y.to(args.device), subject_idxs.to(args.device)
            
            with torch.no_grad():
                class_logits, subject_logits = model(X)
            
            val_loss_class = F.cross_entropy(class_logits, y).item()
            val_loss_subject = F.cross_entropy(subject_logits, subject_idxs).item()
            val_loss_combined = val_loss_class + val_loss_subject
            
            val_loss.append(val_loss_combined)
            val_acc_class.append(accuracy_class(class_logits, y).item())
            val_acc_subject.append(accuracy_subject(subject_logits, subject_idxs).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc class: {np.mean(train_acc_class):.3f} | train acc subject: {np.mean(train_acc_subject):.3f} | val loss: {np.mean(val_loss):.3f} | val acc class: {np.mean(val_acc_class):.3f} | val acc subject: {np.mean(val_acc_subject):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({
                "train_loss": np.mean(train_loss),
                "train_acc_class": np.mean(train_acc_class),
                "train_acc_subject": np.mean(train_acc_subject),
                "val_loss": np.mean(val_loss),
                "val_acc_class": np.mean(val_acc_class),
                "val_acc_subject": np.mean(val_acc_subject)
            })
        
        if np.mean(val_acc_class) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc_class)
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds_class = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Evaluation"):
        X = X.to(args.device)
        class_logits, subject_logits = model(X)
        preds_class.append(class_logits.detach().cpu())
        
    preds_class = torch.cat(preds_class, dim=0).numpy()
    
    np.save(os.path.join(logdir, "submission"), preds_class)
    
    cprint(f"Submission {preds_class.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
