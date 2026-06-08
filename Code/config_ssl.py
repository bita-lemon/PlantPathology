# config_ssl.py
class SSLConfig:
    # مسیر دیتاست
    dataset_path = '/kaggle/input/datasets/nirmalsankalana/cassava-leaf-disease-classification/data'
    
    # تنظیمات مدل
    img_size = 224
    feature_dim = 512
    projection_dim = 128
    
    # تنظیمات آموزش
    batch_size = 32
    epochs = 30
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    temperature = 0.5
    
    # مسیر ذخیره
    save_path = 'ssl_encoder.pth'
    
config = SSLConfig()