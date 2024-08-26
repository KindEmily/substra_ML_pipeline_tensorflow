import numpy as np

def opener(path):
    train_images = np.load(path / 'train_images.npy')
    train_labels = np.load(path / 'train_labels.npy')
    test_images = np.load(path / 'test_images.npy')
    test_labels = np.load(path / 'test_labels.npy')

    return {
        'train': {
            'images': train_images,
            'labels': train_labels
        },
        'test': {
            'images': test_images,
            'labels': test_labels
        }
    }