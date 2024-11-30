import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

def load_contract_data(json_file1, json_file2):
    
    with open(json_file1, 'r') as file:
        contracts1 = json.load(file)

    with open(json_file2, 'r') as file:
        contracts2 = json.load(file)

    all_contracts = contracts1 + contracts2
    return all_contracts

def preprocess_data(contracts):

    source_codes = []
    categories = []
    categories_keywords = {
        "Stablecoins": ["stablecoin", "peg", "usd", "price stability", "reserve"],
        "DeFi": ["liquidity", "stake", "yield", "swap", "borrow", "lending", "apy", "farm"],
        "NFT": ["nft", "mint", "art", "collectible", "auction", "tokenURI"],
        "Governance Tokens": ["vote", "proposal", "governance", "treasury", "delegate"],
        "Utility Tokens": ["utility", "membership", "fee", "access"],
        "Payment Tokens": ["payment", "merchant", "transfer", "settlement"],
        "Gaming/Metaverse": ["game", "play-to-earn", "metaverse", "avatar", "gaming"],
        "Wallet Contracts": ["wallet", "deposit", "withdraw", "key", "account"],
        "Router Contracts": ["swap", "addliquidity", "removeliquidity", "route"],
        "Bridge Contracts": ["bridge", "cross-chain", "wrapped", "transfer"],
        "Standard Tokens (ERC20/BEP20)": ["erc20", "balanceof", "approve", "transferfrom"],
        "Other/Uncategorized": []
    }

    for contract in contracts:
        source_code = contract.get('SourceCode', '').lower()
        abi = contract.get('ABI', '').lower()
        if not source_code and not abi:  
            continue

        source_codes.append(source_code)
        assigned_category = "Other"

        for category, keywords in categories_keywords.items():
            if any(keyword in source_code or keyword in abi for keyword in keywords):
                assigned_category = category
                break

        categories.append(assigned_category)

    label_encoder = LabelEncoder()
    categories_encoded = label_encoder.fit_transform(categories)

    return source_codes, categories_encoded, label_encoder

def vectorize_data(source_codes):

    vectorizer = TfidfVectorizer(
        max_features=5000,  
        stop_words='english', 
        min_df=5,  
        max_df=0.95 
    )
    X = vectorizer.fit_transform(source_codes)
    return X, vectorizer

def train_classifier(X, y):
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Model Performance Metrics:")
    print(f"- Accuracy: {accuracy:.4f}")
    print(f"- Precision: {precision:.4f}")
    print(f"- Recall: {recall:.4f}")
    print(f"- F1 Score: {f1:.4f}")

    print(f"Random Forest accuracy: {accuracy:.4f}")

    return rf_model

def predict_categories(rf_model, X, label_encoder):

    y_pred = rf_model.predict(X)
    predicted_categories = label_encoder.inverse_transform(y_pred)
    return predicted_categories

def save_predictions_to_csv(contracts, predicted_categories, output_file, verified_dict):
    output_data = []
    for contract, predicted_category in zip(contracts, predicted_categories):
        contract_name = contract.get('ContractName', '').lower()
        address = verified_dict.get(contract_name, 'Unknown Address')  
        
        output_data.append((address, predicted_category))

    df = pd.DataFrame(output_data, columns=['address', 'predictedCategory'])
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":

    verified_addresses_file = "verified-contract-address.csv"
    try:
        verified_addresses = pd.read_csv(verified_addresses_file)
        verified_addresses["ContractName"] = verified_addresses["ContractName"].str.lower()
        verified_dict = verified_addresses.set_index("ContractName")["ContractAddress"].to_dict()
        
        print(f"Verified contracts in CSV: {len(verified_dict)}")
    except FileNotFoundError:
        print(f"Error: File {verified_addresses_file} not found.")
        verified_dict = {}
    except Exception as e:
        print(f"An error occurred: {e}")
        verified_dict = {}

    json_file1 = "contract_details_part1.json"
    json_file2 = "contract_details_part2.json"

    contracts = load_contract_data(json_file1, json_file2)
    source_codes, categories_encoded, label_encoder = preprocess_data(contracts)
    X, vectorizer = vectorize_data(source_codes)
    rf_model = train_classifier(X, categories_encoded)
    predicted_categories = predict_categories(rf_model, X, label_encoder)

    save_predictions_to_csv(
        contracts,
        predicted_categories,
        "categorized_contracts_binance_nlp.csv",
        verified_dict
    )
