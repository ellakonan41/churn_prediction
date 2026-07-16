from src.model_trainer import train_model

if __name__ == "__main__":
    print("🚀 Entraînement du modèle...")
    pipeline, X_test, Y_test = train_model()
    print("✅ Modèle entraîné et sauvegardé avec succès!")
    print(f"Accuracy sur le test set: {pipeline.score(X_test, Y_test):.3f}")
