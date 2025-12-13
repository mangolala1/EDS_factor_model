"""
Country to Continent Mapping
Maps country names to 7 continents for factor model
"""
from typing import Dict

# Mapping of countries to 7 continents
COUNTRY_TO_REGION: Dict[str, str] = {

    # North America – Developed
    "United States": "North America - Developed",
    "Canada": "North America - Developed",

    # Latin America – Emerging
    "Mexico": "Latin America - Emerging",

    # Europe – Developed
    "Belgium": "Europe - Developed",
    "United Kingdom": "Europe - Developed",
    "France": "Europe - Developed",
    "Germany": "Europe - Developed",
    "Sweden": "Europe - Developed",
    "Norway": "Europe - Developed",

    # Europe – Emerging
    "Romania": "Europe - Emerging",
    "Bulgaria": "Europe - Emerging",

    # Asia Pacific – Developed
    "Japan": "Asia Pacific - Developed",
    "South Korea": "Asia Pacific - Developed",
    "Hong Kong": "Asia Pacific - Developed",
    "Singapore": "Asia Pacific - Developed",
    "Australia": "Asia Pacific - Developed",
    "Taiwan": "Asia Pacific - Developed",

    # Asia Pacific – Emerging
    "China": "Asia Pacific - Emerging",
    "Philippines": "Asia Pacific - Emerging",
    "Malaysia": "Asia Pacific - Emerging",
    "India": "Asia Pacific - Emerging",
    "Thailand": "Asia Pacific - Emerging",

    # Middle East – Emerging
    "Kuwait": "Middle East - Emerging",
    "Turkey": "Emerging Europe / Eurasia - Emerging",

    # Africa – Emerging
    "Nigeria": "Africa - Emerging",

    # Eurasia – Emerging
    "Russian Federation": "Emerging Europe / Eurasia - Emerging",
}

def get_continent(country: str) -> str:
    """
    Map country name to REGION (continent + development)

    Returns:
        REGION string, e.g. 'Asia Pacific - Developed'
        Returns 'Other - Unknown' if country not found
    """
    if not country:
        return "Other - Unknown"

    try:
        import pandas as pd
        if pd.isna(country):
            return "Other - Unknown"
    except Exception:
        pass

    country_str = str(country).strip()

    # Direct lookup
    if country_str in COUNTRY_TO_REGION:
        return COUNTRY_TO_REGION[country_str]

    # Case-insensitive fallback
    for key, value in COUNTRY_TO_REGION.items():
        if country_str.lower() == key.lower():
            return value

    return "Other - Unknown"

def map_countries_to_continents(country_series) -> list:
    """
    Map a pandas Series of countries to continents
    
    Args:
        country_series: Series of country names
        
    Returns:
        List of continent names
    """
    import pandas as pd
    return [get_continent(country) for country in country_series]

