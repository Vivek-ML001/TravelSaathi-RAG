import pandas as pd


def load_travel_dataset(filepath):
    """Load and clean the travel dataset."""

    df = pd.read_csv(filepath)

    # Remove accidental/empty column
    df = df.drop(columns=[" persona"], errors="ignore")

    # Remove rows with critical missing values
    df = df.dropna(
        subset=[
            "persona",
            "vibe_tags",
            "budget_level",
            "best_season",
            "description",
            "city",
            "country",
            "region",
        ]
    )

    return df


def create_travel_documents(df):
    """Convert each travel record into a text document for RAG."""

    documents = []

    for _, row in df.iterrows():

        document = f"""
Destination: {row['city']}, {row['country']}
Region: {row['region']}
Traveler Persona: {row['persona']}
Budget Level: {row['budget_level']}
Best Season: {row['best_season']}
Travel Vibes: {row['vibe_tags']}

Description:
{row['description']}
""".strip()

        documents.append(document)

    return documents


if __name__ == "__main__":

    filepath = "data/wanderlust_destinations.csv"

    df = load_travel_dataset(filepath)

    documents = create_travel_documents(df)

    print("Dataset rows:", len(df))
    print("Travel documents:", len(documents))

    print("\nSample document:\n")
    print(documents[0])