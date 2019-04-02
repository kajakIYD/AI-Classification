class HeartDiseaseUCIDatarow:

    def __init__(self, datarow):
        self.age = datarow[0]
        self.sex = datarow[1]
        self.cp = datarow[2]
        self.trestbps = datarow[3]
        self.chol = datarow[4]
        self.fbs = datarow[5]
        self.restecg = datarow[6]
        self.thalach = datarow[7]
        self.exang = datarow[8]
        self.oldpeak = datarow[9]
        self.slope = datarow[10]
        self.ca = datarow[11]
        self.thal = datarow[12]
        self.target = datarow[13]
        self.datarow_dict = {'age': self.age, 'sex': self.sex, 'cp': self.cp, 'trestbps': self.trestbps,
                             'chol': self.chol, 'fbs': self.fbs, 'restecg': self.restecg, 'thalach': self.thalach,
                             'exang': self.exang, 'oldpeak': self.oldpeak, 'slope': self.slope, 'ca': self.ca,
                             'thal': self.thal, 'target': self.target}
