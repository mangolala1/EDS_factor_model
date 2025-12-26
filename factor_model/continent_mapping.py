"""
Country to Continent Mapping
Maps country names to 7 continents for factor model
Includes developed/developing classification
"""
from typing import Dict

# Mapping of countries to 7 continents
COUNTRY_TO_CONTINENT: Dict[str, str] = {
    # North America
    'United States': 'North America',
    'USA': 'North America',
    'US': 'North America',
    'Canada': 'North America',
    'Mexico': 'North America',
    
    # South America
    'Brazil': 'South America',
    'Argentina': 'South America',
    'Chile': 'South America',
    'Colombia': 'South America',
    'Peru': 'South America',
    'Venezuela': 'South America',
    
    # Europe
    'United Kingdom': 'Europe',
    'UK': 'Europe',
    'Germany': 'Europe',
    'France': 'Europe',
    'Italy': 'Europe',
    'Spain': 'Europe',
    'Netherlands': 'Europe',
    'Switzerland': 'Europe',
    'Sweden': 'Europe',
    'Norway': 'Europe',
    'Denmark': 'Europe',
    'Belgium': 'Europe',
    'Austria': 'Europe',
    'Ireland': 'Europe',
    'Finland': 'Europe',
    'Poland': 'Europe',
    'Portugal': 'Europe',
    'Greece': 'Europe',
    'Russia': 'Europe',
    
    # Asia
    'China': 'Asia',
    'Japan': 'Asia',
    'India': 'Asia',
    'South Korea': 'Asia',
    'Singapore': 'Asia',
    'Hong Kong': 'Asia',
    'Taiwan': 'Asia',
    'Thailand': 'Asia',
    'Malaysia': 'Asia',
    'Indonesia': 'Asia',
    'Philippines': 'Asia',
    'Vietnam': 'Asia',
    'Australia': 'Asia',  # Often grouped with Asia in finance
    'New Zealand': 'Asia',
    
    # Africa
    'South Africa': 'Africa',
    'Egypt': 'Africa',
    'Nigeria': 'Africa',
    'Kenya': 'Africa',
    'Morocco': 'Africa',
    
    # Oceania (if separate from Asia)
    # 'Australia': 'Oceania',
    # 'New Zealand': 'Oceania',
    
    # Middle East
    'Saudi Arabia': 'Middle East',
    'United Arab Emirates': 'Middle East',
    'UAE': 'Middle East',
    'Israel': 'Middle East',
    'Turkey': 'Middle East',
    'Qatar': 'Middle East',
    'Kuwait': 'Middle East',
}

# Mapping of countries to developed/developing status (MSCI-style classification)
COUNTRY_TO_DEVELOPED_STATUS: Dict[str, str] = {
    # Developed Markets
    'United States': 'Developed',
    'USA': 'Developed',
    'US': 'Developed',
    'Canada': 'Developed',
    'United Kingdom': 'Developed',
    'UK': 'Developed',
    'Germany': 'Developed',
    'France': 'Developed',
    'Italy': 'Developed',
    'Spain': 'Developed',
    'Netherlands': 'Developed',
    'Switzerland': 'Developed',
    'Sweden': 'Developed',
    'Norway': 'Developed',
    'Denmark': 'Developed',
    'Belgium': 'Developed',
    'Austria': 'Developed',
    'Ireland': 'Developed',
    'Finland': 'Developed',
    'Portugal': 'Developed',
    'Greece': 'Developed',
    'Japan': 'Developed',
    'South Korea': 'Developed',
    'Singapore': 'Developed',
    'Hong Kong': 'Developed',
    'Taiwan': 'Developed',
    'Australia': 'Developed',
    'New Zealand': 'Developed',
    'Israel': 'Developed',
    
    # Developing Markets
    'Mexico': 'Developing',
    'Brazil': 'Developing',
    'Argentina': 'Developing',
    'Chile': 'Developing',
    'Colombia': 'Developing',
    'Peru': 'Developing',
    'Venezuela': 'Developing',
    'Poland': 'Developing',
    'Russia': 'Developing',
    'China': 'Developing',
    'India': 'Developing',
    'Thailand': 'Developing',
    'Malaysia': 'Developing',
    'Indonesia': 'Developing',
    'Philippines': 'Developing',
    'Vietnam': 'Developing',
    'South Africa': 'Developing',
    'Egypt': 'Developing',
    'Nigeria': 'Developing',
    'Kenya': 'Developing',
    'Morocco': 'Developing',
    'Saudi Arabia': 'Developing',
    'United Arab Emirates': 'Developing',
    'UAE': 'Developing',
    'Turkey': 'Developing',
    'Qatar': 'Developing',
    'Kuwait': 'Developing',
}

def get_continent(country: str) -> str:
    """
    Map country name to continent
    
    Args:
        country: Country name
        
    Returns:
        Continent name (one of: North America, South America, Europe, Asia, Africa, Middle East, Oceania)
        Returns 'Unknown' if country not found
    """
    if not country:
        return 'Unknown'
    
    # Handle NaN values
    try:
        import pandas as pd
        if pd.isna(country):
            return 'Unknown'
    except:
        pass
    
    country_str = str(country).strip()
    
    # Direct lookup
    if country_str in COUNTRY_TO_CONTINENT:
        return COUNTRY_TO_CONTINENT[country_str]
    
    # Case-insensitive lookup
    for key, value in COUNTRY_TO_CONTINENT.items():
        if country_str.lower() == key.lower():
            return value
    
    # Default to 'Unknown' if not found
    return 'Unknown'

def get_developed_status(country: str) -> str:
    """
    Map country name to developed/developing status
    
    Args:
        country: Country name
        
    Returns:
        'Developed' or 'Developing', or 'Unknown' if country not found
    """
    if not country:
        return 'Unknown'
    
    # Handle NaN values
    try:
        import pandas as pd
        if pd.isna(country):
            return 'Unknown'
    except:
        pass
    
    country_str = str(country).strip()
    
    # Direct lookup
    if country_str in COUNTRY_TO_DEVELOPED_STATUS:
        return COUNTRY_TO_DEVELOPED_STATUS[country_str]
    
    # Case-insensitive lookup
    for key, value in COUNTRY_TO_DEVELOPED_STATUS.items():
        if country_str.lower() == key.lower():
            return value
    
    # Default to 'Unknown' if not found
    return 'Unknown'

def get_continent_developed(country: str) -> str:
    """
    Map country to combined continent and developed/developing classification
    Returns format: "Continent Developed" or "Continent Developing"
    Example: "North America Developed", "Asia Developing"
    
    Args:
        country: Country name
        
    Returns:
        Combined classification string (e.g., "North America Developed")
        Returns "Unknown Unknown" if country not found
    """
    continent = get_continent(country)
    dev_status = get_developed_status(country)
    
    if continent == 'Unknown' or dev_status == 'Unknown':
        return 'Unknown Unknown'
    
    return f"{continent} {dev_status}"

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

def map_countries_to_continent_developed(country_series) -> list:
    """
    Map a pandas Series of countries to combined continent and developed/developing classification
    
    Args:
        country_series: Series of country names
        
    Returns:
        List of combined classifications (e.g., "North America Developed")
    """
    return [get_continent_developed(country) for country in country_series]

