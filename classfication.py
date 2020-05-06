from sklearn.model_selection import cross_val_score

def classify(X,Y,forest):
    

    forest.fit(X,Y)

    score = cross_val_score(forest, X, Y)

    return score
    
