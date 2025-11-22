import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def download_nltk_resources():
    resources = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',
        'stopwords': 'corpora/stopwords', 
        'wordnet': 'corpora/wordnet',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'
    }
    for resource, path in resources.items():
        try:
            nltk.data.find(path)
            print(f"{resource} already downloaded")
        except LookupError:
            print(f"Downloading {resource}...")
            try:
                nltk.download(resource, quiet=True)
                print(f"{resource} downloaded successfully")
            except Exception as e:
                print(f"Failed to download {resource}: {e}")

try:
    download_nltk_resources()
except Exception as e:
    print(f"NLTK download warning: {e}")
    print("The app will continue with basic functionality")

class NLPCategorizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
            print("Using empty stopwords set")

        self.category_keywords = {
            'Salary': [
                'salary', 'paycheck', 'payroll', 'wage', 'stipend', 'earnings',
                'compensation', 'income', 'payment', 'deposit'
            ],
            'Freelance': [
                'freelance', 'contract', 'consulting', 'gig', 'project', 'client',
                'contractor', 'independent', 'self-employed','freelancing'
            ],
            'Business': [
                'business', 'revenue', 'sales', 'profit', 'enterprise', 'venture',
                'company', 'corporation', 'firm'
            ],
            'Investments': [
                'dividend', 'interest', 'investment', 'stock', 'mutual', 'fund',
                'return', 'portfolio', 'yield', 'capital', 'gain'
            ],
            'Rental': [
                'rent', 'rental', 'property', 'lease', 'tenant', 'landlord',
                'apartment', 'house', 'real estate'
            ],
            'Groceries': [
                'grocery', 'supermarket', 'food', 'walmart', 'kroger', 'aldi',
                'costco', 'vegetable', 'fruit', 'produce', 'meat', 'dairy',
                'bread', 'milk', 'egg', 'rice', 'pasta', 'snack', 'beverage',
                'kitchen', 'purchase', 'buy', 'shop', 'market','groceries'
            ],
            'Dining': [
                'restaurant', 'cafe', 'coffee', 'starbucks', 'mcdonald', 'kfc',
                'pizza', 'dining', 'eat', 'food', 'lunch', 'dinner', 'breakfast',
                'meal', 'burger', 'sandwich', 'takeaway', 'takeout'
            ],
            'Transport': [
                'uber', 'lyft', 'taxi', 'transport', 'fuel', 'gas', 'petrol',
                'bus', 'train', 'metro', 'subway', 'transit', 'commute',
                'travel', 'fare', 'ticket', 'airport', 'flight'
            ],
            'Entertainment': [
                'movie', 'netflix', 'spotify', 'entertainment', 'game', 'concert',
                'theater', 'cinema', 'music', 'streaming', 'subscription',
                'hobby', 'leisure', 'fun', 'amusement'
            ],
            'Shopping': [
                'amazon', 'mall', 'sale' ,'shopping', 'store', 'purchase', 'buy',
                'order', 'clothes', 'electronic', 'fashion', 'retail',
                'merchandise', 'product', 'item'
            ],
            'Bills': [
                'bill', 'electricity', 'water', 'internet', 'phone', 'utility',
                'subscription', 'fee', 'charge', 'payment', 'service',
                'maintenance', 'upkeep'
            ],
            'Healthcare': [
                'hospital', 'doctor', 'medical', 'pharmacy', 'health', 'clinic',
                'dental', 'medicine', 'treatment', 'therapy', 'insurance',
                'wellness', 'checkup'
            ]
        }
        
        self.income_keywords = [
            'salary', 'credited', 'deposit', 'income', 'payment received',
            'refund', 'reimbursement', 'bonus', 'dividend', 'interest',
            'transfer in', 'gift received', 'freelancing', 'consulting',
            'payment from', 'received from', 'paycheck', 'stipend',
            'royalty', 'rent received', 'investment return', 'credit',
            'earnings', 'revenue', 'allowance', 'scholarship',
            'commission', 'profit', 'gain', 'winning', 'lottery', 'grant',
            'tax return','freelance','contract'
        ]
        
        self.expense_keywords = [
            'purchase', 'payment', 'bill', 'subscription', 'fee',
            'shopping', 'grocery', 'restaurant', 'food', 'dining',
            'transport', 'uber', 'taxi', 'fuel', 'gas',
            'entertainment', 'movie', 'netflix', 'spotify',
            'rent', 'mortgage', 'utility', 'electricity', 'water',
            'insurance', 'medical', 'doctor', 'pharmacy',
            'withdrawal', 'transfer out', 'sent to', 'paid to', 'debit',
            'spent', 'expense', 'cost', 'charge', 'bought', 'buy', 'paid for'
        ]

    def preprocess_text(self, text):
        if not text:
            return []
        
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        try:
            tokens = word_tokenize(text)
            tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
            lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            return lemmatized_tokens
            
        except Exception as e:
            print(f"NLTK tokenization failed, using fallback: {e}")
            tokens = text.split()
            tokens = [token for token in tokens if len(token) > 2]
            return tokens

    def detect_transaction_type(self, description):
        """Enhanced transaction type detection"""
        if not description:
            return "expense"
        
        desc_lower = description.lower()
        
        # Strong income indicators
        strong_income_indicators = [
            'salary', 'deposit', 'income', 'credited', 'refund', 'payment received',
            'freelance', 'contract', 'project payment', 'reimbursement', 'bonus',
            'tax refund', 'dividend', 'interest', 'investment return','freelancing','fixed deposit','fd',
        ]
        
        # Check for strong income patterns
        for indicator in strong_income_indicators:
            if indicator in desc_lower:
                return "income"
        
        # Strong expense indicators (especially for groceries)
        strong_expense_indicators = [
            'grocery', 'supermarket', 'shopping', 'purchase', 'bill', 'payment',
            'restaurant', 'cafe', 'coffee', 'food', 'buy', 'fuel', 'gas'
        ]
        
        for indicator in strong_expense_indicators:
            if indicator in desc_lower:
                return "expense"
        
        # Default to expense for safety
        return "expense"

    def categorize_transaction(self, description):
        """Enhanced categorization with better pattern matching"""
        if not description:
            return "Other"
        
        desc_lower = description.lower()
        
        # Groceries and Food
        if any(word in desc_lower for word in ['grocery', 'supermarket', 'food', 'vegetable', 'fruit', 'meat',
        'groceries','provision','bread','rice','dmart']):
            return "Groceries"
        
        # Dining
        elif any(word in desc_lower for word in ['restaurant', 'cafe', 'coffee', 'dining', 'eat out']):
            return "Dining"
        
        # Transport
        elif any(word in desc_lower for word in ['fuel', 'gas', 'petrol', 'transport', 'bus', 'train'
        'ola','uber','taxi','metro','diesel','parking','rapido']):
            return "Transport"
        
        # Bills
        elif any(word in desc_lower for word in ['bill', 'electricity', 'water', 'internet', 'mobile',
        'subscription','wifi',]):
            return "Bills"
        
        # Entertainment
        elif any(word in desc_lower for word in ['movie', 'movies', 'tickets', 'cinema', 'entertainment',
        'show', 'game','concert','event','theatre']):
            return "Entertainment"
        
        # Shopping
        elif any(word in desc_lower for word in ['shopping', 'mall', 'store', 'purchase']):
            return "Shopping"
        
        # Healthcare
        elif any(word in desc_lower for word in ['medical', 'health', 'doctor', 'pharmacy','medicine','pharmacy',
        'lab','health check','tablet']):
            return "Healthcare"
        
        # Income categories
        elif any(word in desc_lower for word in ['salary', 'paycheck']):
            return "Salary"
        elif any(word in desc_lower for word in ['freelance', 'contract']):
            return "Freelance"
        elif any(word in desc_lower for word in ['investment', 'dividend']):
            return "Investments"
        
        # Default
        else:
            return "Other"

    def detect_transaction_type_fallback(self, description):
        """Simple fallback for transaction type detection"""
        if not description:
            return "expense"
        desc_lower = description.lower()
        
        income_indicators = ['salary', 'deposit', 'income', 'credited', 'refund','fixed deposit','fd']
        for indicator in income_indicators:
            if indicator in desc_lower:
                return "income"
        return "expense"

    def categorize_transaction_fallback(self, description):
        """Simple fallback for categorization"""
        if not description:
            return "Other"
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ['grocery', 'food', 'supermarket']):
            return "Groceries"
        elif any(word in desc_lower for word in ['restaurant', 'cafe', 'coffee']):
            return "Dining"
        elif any(word in desc_lower for word in ['salary', 'paycheck']):
            return "Salary"
        else:
            return "Other"


nlp_categorizer = NLPCategorizer()

def detect_transaction_type(description):
    return nlp_categorizer.detect_transaction_type(description)

def categorize_transaction(description):
    return nlp_categorizer.categorize_transaction(description)

def manual_categorization(transaction_id, new_category, user_email):
    try:
        from utils.database import load_data, save_data
        
        data = load_data()
        
        if "transactions" not in data:
            return False, "No transactions found in database"
        
        if user_email not in data["transactions"]:
            return False, "User not found"
        
        transaction_updated = False
        for transaction in data["transactions"][user_email]:
            if str(transaction.get("id", "")) == str(transaction_id):
                transaction["category"] = new_category
                transaction_updated = True
                break
        
        if transaction_updated:
            save_data(data)
            return True, "Category updated successfully"
        else:
            return False, "Transaction not found"
            
    except Exception as e:
        return False, f"Error updating category: {str(e)}"

CATEGORIES = {
    'Salary': 'Regular employment income',
    'Freelance': 'Freelance or contract work',
    'Business': 'Business revenue and profits', 
    'Investments': 'Investment returns and dividends',
    'Rental': 'Rental property income',
    'Groceries': 'Food and household items',
    'Dining': 'Restaurants and eating out',
    'Transport': 'Transportation costs',
    'Entertainment': 'Entertainment and leisure',
    'Shopping': 'Shopping and retail',
    'Bills': 'Bills and utilities',
    'Healthcare': 'Medical and healthcare',
    'Other': 'Other expenses'
}