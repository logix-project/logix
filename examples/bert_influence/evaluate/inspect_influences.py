import torch

from examples.glue.pipeline import get_loaders


def main(data_name: str = "sst2", model_id: int = 0):
    scores = torch.load(
        f"../files/results/{model_id}/{data_name}_if.pt", map_location="cpu"
    )

    _, eval_train_loader, valid_loader = get_loaders(
        data_name=data_name,
        train_indices=None,
        valid_indices=list(range(128)),
        eval_batch_size=8,
    )

    for i in range(16):
        print("=" * 80)
        print(f"{i}th data point")
        print(f"Sequence: {valid_loader.dataset[i]['sentence']}")
        print(f"Label: {valid_loader.dataset[i]['label']}")

        print("Most influential data point")
        rank = torch.argsort(scores[i], descending=True)
        for j in range(3):
            print(f"Rank {j} (score = {scores[i][rank[j]]})")
            print(f"Sentence: {eval_train_loader.dataset[int(rank[j])]['sentence']}")
            print(f"Label: {eval_train_loader.dataset[int(rank[j])]['label']}")


if __name__ == "__main__":
    main()
