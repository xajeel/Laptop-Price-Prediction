import pickle as pk

def save_object(save_path, best_model):
    """
    This function Saves the Object in the Folder
    """
    with open(save_path, 'wb') as file:
        pk.dump(best_model, file)
