import numpy as np
import sklearn.metrics


class Scores:
    def __init__(self):
        self.store = []

    def update(self, labels, predictions):

        l1, l2, l3 = labels
        p1, p2, p3 = predictions

        l1, l2, l3 = (
            l1.detach().cpu().numpy(),
            l2.detach().cpu().numpy(),
            l3.detach().cpu().numpy(),
        )
        p1, p2, p3 = (
            p1.detach().cpu().numpy(),
            p2.detach().cpu().numpy(),
            p3.detach().cpu().numpy(),
        )

        p1 = np.argmax(p1, axis=1)
        p2 = np.argmax(p2, axis=1)
        p3 = np.argmax(p3, axis=1)

        temp_store = []
        temp_store.append(sklearn.metrics.recall_score(l1, p1, average="macro"))
        temp_store.append(sklearn.metrics.recall_score(l2, p2, average="macro"))
        temp_store.append(sklearn.metrics.recall_score(l3, p3, average="macro"))

        self.store.append(np.average(temp_store, weights=[2, 1, 1]))

    def get_metrics(self):
        return np.mean(self.store)


def epoch_score(loss, scr):
    epoch_score = scr.get_metrics()
    print(f"Loss: {loss:.4f} | Recall_Score: {epoch_score:.4f}")
