from sklearn.preprocessing import StandardScaler, LabelEncoder

def scale_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

def encode_labels(labels):
    encoder = LabelEncoder()
    return encoder.fit_transform(labels)
