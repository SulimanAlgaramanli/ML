    for model_name, model in self.models.items():
            X = self.data['text']
            y = self.data['label']
            model.train(X, y)  # استخدام كل البيانات للتدريب
            X_train = model.vectorizer.fit_transform(X)  # تحويل النصوص إلى تمثيل رقمي
            self.plot_model(X_train, y, model_name)
