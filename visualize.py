import matplotlib.pyplot as plt


def visualize_history(history):
    # summarize history for accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accurracy.png', bbox_inches='tight')
    plt.show()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    with open('history/history.db', 'rb') as file_pi:
        history = pickle.load(history.history, file_pi)
        visualize_history(history)
