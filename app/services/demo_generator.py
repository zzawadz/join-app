"""
Demo data generator for creating synthetic test datasets.
Supports both people/customer and company data for record linkage demos.
"""
import random
import os
import csv
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd

try:
    from faker import Faker
except ImportError:
    Faker = None


# Common nickname mappings for realistic variations
NICKNAME_MAP = {
    "Robert": ["Bob", "Rob", "Bobby", "Robbie"],
    "William": ["Will", "Bill", "Billy", "Willy"],
    "Richard": ["Rich", "Rick", "Dick", "Ricky"],
    "James": ["Jim", "Jimmy", "Jamie"],
    "Michael": ["Mike", "Mikey", "Mick"],
    "Elizabeth": ["Liz", "Beth", "Lizzy", "Betty", "Eliza"],
    "Jennifer": ["Jen", "Jenny", "Jenn"],
    "Katherine": ["Kate", "Katie", "Kathy", "Kat"],
    "Margaret": ["Maggie", "Meg", "Peggy", "Marge"],
    "Patricia": ["Pat", "Patty", "Trish"],
    "Christopher": ["Chris", "Topher"],
    "Nicholas": ["Nick", "Nicky"],
    "Alexander": ["Alex", "Xander"],
    "Benjamin": ["Ben", "Benny"],
    "Jonathan": ["Jon", "Johnny"],
    "Thomas": ["Tom", "Tommy"],
    "Edward": ["Ed", "Eddie", "Ted"],
    "Anthony": ["Tony", "Ant"],
    "Joseph": ["Joe", "Joey"],
    "Daniel": ["Dan", "Danny"],
    "Matthew": ["Matt", "Matty"],
    "Andrew": ["Andy", "Drew"],
    "Steven": ["Steve", "Stevie"],
    "Charles": ["Charlie", "Chuck"],
}

# Address abbreviations
ADDRESS_ABBREVS = {
    "Street": ["St", "St.", "Str"],
    "Avenue": ["Ave", "Ave.", "Av"],
    "Boulevard": ["Blvd", "Blvd.", "Bl"],
    "Drive": ["Dr", "Dr.", "Drv"],
    "Lane": ["Ln", "Ln.", "La"],
    "Road": ["Rd", "Rd."],
    "Court": ["Ct", "Ct."],
    "Place": ["Pl", "Pl."],
    "Circle": ["Cir", "Cir."],
}

# State abbreviations
STATE_ABBREVS = {
    "California": "CA", "Texas": "TX", "Florida": "FL", "New York": "NY",
    "Pennsylvania": "PA", "Illinois": "IL", "Ohio": "OH", "Georgia": "GA",
    "North Carolina": "NC", "Michigan": "MI", "New Jersey": "NJ",
    "Virginia": "VA", "Washington": "WA", "Arizona": "AZ", "Massachusetts": "MA",
    "Tennessee": "TN", "Indiana": "IN", "Missouri": "MO", "Maryland": "MD",
    "Wisconsin": "WI", "Colorado": "CO", "Minnesota": "MN", "South Carolina": "SC",
    "Alabama": "AL", "Louisiana": "LA", "Kentucky": "KY", "Oregon": "OR",
    "Oklahoma": "OK", "Connecticut": "CT", "Utah": "UT", "Iowa": "IA",
}

# Company suffixes
COMPANY_SUFFIXES = ["Inc", "Inc.", "Incorporated", "LLC", "L.L.C.", "Corp", "Corp.", "Corporation", "Co", "Co."]

# Industry synonyms
INDUSTRY_SYNONYMS = {
    "Technology": ["Tech", "IT", "Information Technology", "Software"],
    "Healthcare": ["Health", "Medical", "Medicine", "Health Care"],
    "Finance": ["Financial", "Financial Services", "Banking"],
    "Manufacturing": ["Mfg", "Production", "Industrial"],
    "Retail": ["Retail Trade", "Commerce", "Sales"],
    "Education": ["Educational", "Academic", "Training"],
}


class DemoDataGenerator:
    """Generates synthetic demo data for record linkage projects."""

    def __init__(self, seed: int = 42):
        """Initialize generator with a seed for reproducibility."""
        self.seed = seed
        self.rng = random.Random(seed)

        if Faker is None:
            raise ImportError("Faker library is required for demo data generation. Install with: pip install faker")

        self.fake = Faker()
        Faker.seed(seed)

    def _apply_typo(self, text: str, probability: float = 0.1) -> str:
        """Apply random typo to text."""
        if not text or self.rng.random() > probability:
            return text

        text = str(text)
        if len(text) < 2:
            return text

        typo_type = self.rng.choice(["swap", "delete", "double", "replace"])

        if typo_type == "swap" and len(text) > 2:
            # Swap two adjacent characters
            idx = self.rng.randint(0, len(text) - 2)
            return text[:idx] + text[idx + 1] + text[idx] + text[idx + 2:]
        elif typo_type == "delete" and len(text) > 3:
            # Delete a character
            idx = self.rng.randint(1, len(text) - 2)
            return text[:idx] + text[idx + 1:]
        elif typo_type == "double":
            # Double a character
            idx = self.rng.randint(0, len(text) - 1)
            return text[:idx] + text[idx] + text[idx:]
        elif typo_type == "replace":
            # Replace with adjacent key (simplified)
            idx = self.rng.randint(0, len(text) - 1)
            adjacent = {"a": "s", "s": "d", "d": "f", "e": "r", "r": "t", "i": "o", "o": "p", "n": "m", "m": "n"}
            char = text[idx].lower()
            if char in adjacent:
                new_char = adjacent[char]
                if text[idx].isupper():
                    new_char = new_char.upper()
                return text[:idx] + new_char + text[idx + 1:]
        return text

    def _get_nickname(self, first_name: str) -> str:
        """Get a nickname variation for a first name."""
        if first_name in NICKNAME_MAP:
            return self.rng.choice(NICKNAME_MAP[first_name])
        return first_name

    def _vary_address(self, address: str) -> str:
        """Apply variations to address (abbreviations)."""
        for full, abbrevs in ADDRESS_ABBREVS.items():
            if full in address:
                if self.rng.random() > 0.5:
                    address = address.replace(full, self.rng.choice(abbrevs))
        return address

    def _vary_state(self, state: str) -> str:
        """Vary state between full name and abbreviation."""
        if state in STATE_ABBREVS and self.rng.random() > 0.5:
            return STATE_ABBREVS[state]
        # Reverse lookup
        for full, abbrev in STATE_ABBREVS.items():
            if state == abbrev and self.rng.random() > 0.5:
                return full
        return state

    def _vary_phone(self, phone: str) -> str:
        """Apply phone format variations."""
        # Extract digits
        digits = ''.join(c for c in phone if c.isdigit())
        if len(digits) != 10:
            return phone

        formats = [
            f"({digits[:3]}) {digits[3:6]}-{digits[6:]}",
            f"{digits[:3]}-{digits[3:6]}-{digits[6:]}",
            f"{digits[:3]}.{digits[3:6]}.{digits[6:]}",
            f"{digits[:3]} {digits[3:6]} {digits[6:]}",
            digits,
        ]
        return self.rng.choice(formats)

    def _vary_email(self, email: str, first_name: str, last_name: str) -> str:
        """Apply email variations."""
        domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "mail.com"]
        variations = [
            f"{first_name.lower()}.{last_name.lower()}@{self.rng.choice(domains)}",
            f"{first_name.lower()}{last_name.lower()}@{self.rng.choice(domains)}",
            f"{first_name[0].lower()}{last_name.lower()}@{self.rng.choice(domains)}",
            f"{first_name.lower()}_{last_name.lower()}@{self.rng.choice(domains)}",
            f"{last_name.lower()}.{first_name.lower()}@{self.rng.choice(domains)}",
        ]
        return self.rng.choice(variations)

    def _vary_company_name(self, name: str) -> str:
        """Apply company name variations."""
        # Remove/change suffix
        for suffix in COMPANY_SUFFIXES:
            if name.endswith(suffix):
                base = name[:-len(suffix)].strip()
                if self.rng.random() > 0.5:
                    return base + " " + self.rng.choice(COMPANY_SUFFIXES)
                return base
        # Add suffix
        if self.rng.random() > 0.7:
            return name + " " + self.rng.choice(COMPANY_SUFFIXES)
        return name

    def generate_people_dataset(self, n_records: int = 100, seed_offset: int = 0) -> pd.DataFrame:
        """Generate synthetic people/customer records."""
        Faker.seed(self.seed + seed_offset)
        self.fake = Faker()

        records = []
        for i in range(n_records):
            # Choose a name that might have nickname variations
            if self.rng.random() > 0.7:
                first_name = self.rng.choice(list(NICKNAME_MAP.keys()))
            else:
                first_name = self.fake.first_name()

            last_name = self.fake.last_name()
            state = self.rng.choice(list(STATE_ABBREVS.keys()))

            record = {
                "id": i + 1,
                "first_name": first_name,
                "last_name": last_name,
                "email": f"{first_name.lower()}.{last_name.lower()}@{self.fake.free_email_domain()}",
                "phone": self.fake.phone_number()[:14],
                "address": self.fake.street_address(),
                "city": self.fake.city(),
                "state": state,
                "zip": self.fake.zipcode()[:5],
                "date_of_birth": self.fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%Y-%m-%d"),
            }
            records.append(record)

        return pd.DataFrame(records)

    def generate_companies_dataset(self, n_records: int = 100, seed_offset: int = 0) -> pd.DataFrame:
        """Generate synthetic company records."""
        Faker.seed(self.seed + seed_offset)
        self.fake = Faker()

        industries = list(INDUSTRY_SYNONYMS.keys())
        records = []

        for i in range(n_records):
            state = self.rng.choice(list(STATE_ABBREVS.keys()))
            industry = self.rng.choice(industries)

            record = {
                "id": i + 1,
                "company_name": self.fake.company(),
                "industry": industry,
                "address": self.fake.street_address(),
                "city": self.fake.city(),
                "state": state,
                "zip": self.fake.zipcode()[:5],
                "phone": self.fake.phone_number()[:14],
                "website": self.fake.domain_name(),
                "employee_count": self.rng.randint(10, 10000),
            }
            records.append(record)

        return pd.DataFrame(records)

    def create_duplicate_with_variations(self, record: Dict, domain: str = "people") -> Dict:
        """Create a duplicate record with realistic variations."""
        dup = record.copy()

        if domain == "people":
            # First name variation (nickname or typo)
            if self.rng.random() > 0.5:
                dup["first_name"] = self._get_nickname(record["first_name"])
            elif self.rng.random() > 0.7:
                dup["first_name"] = self._apply_typo(record["first_name"], 0.8)

            # Last name typo
            if self.rng.random() > 0.8:
                dup["last_name"] = self._apply_typo(record["last_name"], 0.8)

            # Email variation
            if self.rng.random() > 0.6:
                dup["email"] = self._vary_email(record["email"], dup["first_name"], dup["last_name"])

            # Phone format variation
            if self.rng.random() > 0.5:
                dup["phone"] = self._vary_phone(record["phone"])

            # Address abbreviation
            if self.rng.random() > 0.5:
                dup["address"] = self._vary_address(record["address"])

            # State variation
            if self.rng.random() > 0.5:
                dup["state"] = self._vary_state(record["state"])

            # City typo
            if self.rng.random() > 0.85:
                dup["city"] = self._apply_typo(record["city"], 0.8)

            # Zip transposition
            if self.rng.random() > 0.9 and len(record["zip"]) >= 4:
                z = record["zip"]
                idx = self.rng.randint(0, len(z) - 2)
                dup["zip"] = z[:idx] + z[idx + 1] + z[idx] + z[idx + 2:]

        else:  # companies
            # Company name variation
            if self.rng.random() > 0.4:
                dup["company_name"] = self._vary_company_name(record["company_name"])
            elif self.rng.random() > 0.8:
                dup["company_name"] = self._apply_typo(record["company_name"], 0.8)

            # Industry synonym
            if record["industry"] in INDUSTRY_SYNONYMS and self.rng.random() > 0.5:
                dup["industry"] = self.rng.choice(INDUSTRY_SYNONYMS[record["industry"]])

            # Address abbreviation
            if self.rng.random() > 0.5:
                dup["address"] = self._vary_address(record["address"])

            # State variation
            if self.rng.random() > 0.5:
                dup["state"] = self._vary_state(record["state"])

            # Phone format
            if self.rng.random() > 0.5:
                dup["phone"] = self._vary_phone(record["phone"])

            # Website variation (www prefix)
            if self.rng.random() > 0.5:
                if record["website"].startswith("www."):
                    dup["website"] = record["website"][4:]
                else:
                    dup["website"] = "www." + record["website"]

            # Employee count rounding
            if self.rng.random() > 0.6:
                emp = record["employee_count"]
                if emp > 100:
                    dup["employee_count"] = round(emp, -2)  # Round to nearest 100
                elif emp > 10:
                    dup["employee_count"] = round(emp, -1)  # Round to nearest 10

        return dup

    def create_linkage_datasets(
        self,
        domain: str = "people",
        source_size: int = 100,
        target_size: int = 120,
        overlap_rate: float = 0.4,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create source and target datasets with controlled overlap for record linkage.

        Args:
            domain: "people" or "companies"
            source_size: Number of records in source dataset
            target_size: Number of records in target dataset
            overlap_rate: Proportion of source records that appear (with variations) in target

        Returns:
            Tuple of (source_df, target_df)
        """
        # Generate source dataset
        if domain == "people":
            source_df = self.generate_people_dataset(source_size, seed_offset=0)
        else:
            source_df = self.generate_companies_dataset(source_size, seed_offset=0)

        # Calculate overlapping records
        n_overlap = int(source_size * overlap_rate)
        overlap_indices = self.rng.sample(range(source_size), n_overlap)

        # Generate target dataset base (non-overlapping part)
        n_new = target_size - n_overlap
        if domain == "people":
            new_df = self.generate_people_dataset(n_new, seed_offset=10000)
        else:
            new_df = self.generate_companies_dataset(n_new, seed_offset=10000)

        # Create overlapping records with variations
        overlap_records = []
        for idx in overlap_indices:
            original = source_df.iloc[idx].to_dict()
            varied = self.create_duplicate_with_variations(original, domain)
            overlap_records.append(varied)

        overlap_df = pd.DataFrame(overlap_records)

        # Combine and shuffle target
        target_df = pd.concat([new_df, overlap_df], ignore_index=True)
        target_df = target_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        # Reassign IDs
        target_df["id"] = range(1, len(target_df) + 1)

        return source_df, target_df

    def create_dedup_dataset(
        self,
        domain: str = "people",
        n_unique: int = 100,
        duplicate_rate: float = 0.3,
    ) -> pd.DataFrame:
        """
        Create a dataset with duplicates for deduplication.

        Args:
            domain: "people" or "companies"
            n_unique: Number of unique records
            duplicate_rate: Proportion of records that have duplicates

        Returns:
            DataFrame with duplicates
        """
        # Generate base records
        if domain == "people":
            base_df = self.generate_people_dataset(n_unique)
        else:
            base_df = self.generate_companies_dataset(n_unique)

        # Select records to duplicate
        n_to_duplicate = int(n_unique * duplicate_rate)
        dup_indices = self.rng.sample(range(n_unique), n_to_duplicate)

        # Create duplicates with variations
        dup_records = []
        for idx in dup_indices:
            original = base_df.iloc[idx].to_dict()
            # Each record can have 1-3 duplicates
            n_dups = self.rng.randint(1, 3)
            for _ in range(n_dups):
                varied = self.create_duplicate_with_variations(original, domain)
                dup_records.append(varied)

        dup_df = pd.DataFrame(dup_records)

        # Combine and shuffle
        result_df = pd.concat([base_df, dup_df], ignore_index=True)
        result_df = result_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        # Reassign IDs
        result_df["id"] = range(1, len(result_df) + 1)

        return result_df

    def get_column_mappings(self, domain: str = "people") -> Dict[str, str]:
        """Get default column mappings for a domain."""
        if domain == "people":
            return {
                "first_name": "first_name",
                "last_name": "last_name",
                "email": "email",
                "phone": "phone",
                "address": "address",
                "city": "city",
                "state": "state",
                "zip": "zip",
            }
        else:
            return {
                "company_name": "company_name",
                "industry": "industry",
                "address": "address",
                "city": "city",
                "state": "state",
                "zip": "zip",
                "phone": "phone",
            }

    def get_comparison_config(self, domain: str = "people") -> Dict[str, Dict]:
        """Get default comparison config for a domain."""
        if domain == "people":
            return {
                "first_name": {"method": "jaro_winkler", "threshold": 0.8},
                "last_name": {"method": "jaro_winkler", "threshold": 0.85},
                "email": {"method": "levenshtein", "threshold": 0.7},
                "phone": {"method": "exact", "threshold": 1.0},
                "address": {"method": "jaro_winkler", "threshold": 0.75},
                "city": {"method": "jaro_winkler", "threshold": 0.85},
                "state": {"method": "exact", "threshold": 1.0},
                "zip": {"method": "exact", "threshold": 1.0},
            }
        else:
            return {
                "company_name": {"method": "jaro_winkler", "threshold": 0.8},
                "industry": {"method": "jaro_winkler", "threshold": 0.7},
                "address": {"method": "jaro_winkler", "threshold": 0.75},
                "city": {"method": "jaro_winkler", "threshold": 0.85},
                "state": {"method": "exact", "threshold": 1.0},
                "zip": {"method": "exact", "threshold": 1.0},
                "phone": {"method": "exact", "threshold": 1.0},
            }

    def generate_labeled_pairs(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        domain: str,
        n_matches: int = 15,
        n_non_matches: int = 15,
    ) -> List[Dict]:
        """
        Generate pre-labeled pairs for immediate model training.

        Returns list of dicts with: left_record, right_record, comparison_vector, label
        """
        from app.core.linkage.comparators import compare_records

        pairs = []
        column_mappings = self.get_column_mappings(domain)
        comparison_config = self.get_comparison_config(domain)

        # Find likely matches (high similarity pairs)
        match_pairs = []
        for i in range(min(50, len(source_df))):
            source_record = source_df.iloc[i]
            for j in range(min(50, len(target_df))):
                target_record = target_df.iloc[j]

                # Quick filter: check if names are similar
                if domain == "people":
                    src_name = str(source_record.get("first_name", "")).lower()
                    tgt_name = str(target_record.get("first_name", "")).lower()
                    if src_name[:3] == tgt_name[:3] or src_name in NICKNAME_MAP.get(tgt_name.title(), [src_name]):
                        vector = compare_records(
                            source_record.to_dict(),
                            target_record.to_dict(),
                            column_mappings,
                            comparison_config
                        )
                        avg_score = sum(vector.values()) / len(vector) if vector else 0
                        if avg_score > 0.6:
                            match_pairs.append((i, j, vector, avg_score))
                else:
                    src_name = str(source_record.get("company_name", "")).lower()[:10]
                    tgt_name = str(target_record.get("company_name", "")).lower()[:10]
                    if src_name[:5] == tgt_name[:5]:
                        vector = compare_records(
                            source_record.to_dict(),
                            target_record.to_dict(),
                            column_mappings,
                            comparison_config
                        )
                        avg_score = sum(vector.values()) / len(vector) if vector else 0
                        if avg_score > 0.6:
                            match_pairs.append((i, j, vector, avg_score))

        # Sort by score and take top matches
        match_pairs.sort(key=lambda x: x[3], reverse=True)
        for i, j, vector, _ in match_pairs[:n_matches]:
            pairs.append({
                "left_record": {**source_df.iloc[i].to_dict(), "_idx": int(i)},
                "right_record": {**target_df.iloc[j].to_dict(), "_idx": int(j)},
                "comparison_vector": vector,
                "label": "match"
            })

        # Generate non-matches (random pairs with low similarity)
        used_pairs = set((p["left_record"]["_idx"], p["right_record"]["_idx"]) for p in pairs)
        non_match_attempts = 0

        while len([p for p in pairs if p["label"] == "non_match"]) < n_non_matches and non_match_attempts < 200:
            i = self.rng.randint(0, len(source_df) - 1)
            j = self.rng.randint(0, len(target_df) - 1)

            if (i, j) not in used_pairs:
                vector = compare_records(
                    source_df.iloc[i].to_dict(),
                    target_df.iloc[j].to_dict(),
                    column_mappings,
                    comparison_config
                )
                avg_score = sum(vector.values()) / len(vector) if vector else 0

                # Only add if clearly a non-match
                if avg_score < 0.4:
                    pairs.append({
                        "left_record": {**source_df.iloc[i].to_dict(), "_idx": int(i)},
                        "right_record": {**target_df.iloc[j].to_dict(), "_idx": int(j)},
                        "comparison_vector": vector,
                        "label": "non_match"
                    })
                    used_pairs.add((i, j))

            non_match_attempts += 1

        return pairs


def save_demo_datasets(
    storage_path: str,
    project_id: int,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame = None,
    is_dedup: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Save demo datasets to storage directory.

    Returns tuple of (source_path, target_path) or (dedupe_path, None)
    """
    demo_dir = os.path.join(storage_path, "demo", str(project_id))
    os.makedirs(demo_dir, exist_ok=True)

    if is_dedup:
        dedupe_path = os.path.join(demo_dir, "dedupe_dataset.csv")
        source_df.to_csv(dedupe_path, index=False)
        return dedupe_path, None
    else:
        source_path = os.path.join(demo_dir, "source_dataset.csv")
        target_path = os.path.join(demo_dir, "target_dataset.csv")
        source_df.to_csv(source_path, index=False)
        target_df.to_csv(target_path, index=False)
        return source_path, target_path
