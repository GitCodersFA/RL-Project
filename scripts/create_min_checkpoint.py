from models.feature_extractor import DefectClassifier
import torch, os

if __name__ == '__main__':
    model = DefectClassifier(pretrained=False)
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, 'checkpoints/best_classifier.pth')
    print('Saved minimal classifier checkpoint')
