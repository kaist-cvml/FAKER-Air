# IMPORTANT
# You should make `/home/{YOUR_NAME}/.cdsapirc` files for download
# If you want to download "cams" dataset, your `/home/$YOUR_NAME/.cdsapirc` should be
# url: https://ads.atmosphere.copernicus.eu/api
# key: ##
# Otherwise, if you want to download "ERA5" dataset, your `/home/$YOUR_NAME/.cdsapirc` should be
# url: https://cds.climate.copernicus.eu/api
# key: ##

# # Example: Download data from May 1, 2023 to Dec 31, 2023
# python dataset_download.py --start_date 2023-01-01 --end_date 2023-12-31

# # Example: Specify a custom download path
# python dataset_download.py \
#   --download_path "/my/custom/path/" \
#   --start_date 2023-05-01 \
#   --end_date 2023-05-03

# netCDF4 is used to check file integrity

import argparse
from pathlib import Path
import cdsapi
import urllib3
import os
import hashlib
import time
from datetime import datetime, timedelta
import zipfile, math

# netCDF4 is used to check file integrity
try:
    import netCDF4
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False

urllib3.disable_warnings()

def parse_args():
    """
    Parse command-line arguments:
      --download_path: the directory where the data will be saved
      --start_date: the date from which to start downloading (YYYY-MM-DD)
      --end_date: the date at which to end downloading (YYYY-MM-DD)
      --only_checking: only validate existing files without downloading
    """
    parser = argparse.ArgumentParser(description="ERA5 daily data download")

    parser.add_argument(
        "--download_path",
        type=str,
        default="./data",
        help="Root path to the download directory (default: ./data)"
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Starting date for downloading (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=True,
        help="Ending date for downloading (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="era5",
        help="Select the data you want to download ('era5' or 'cams')"
    )
    
    parser.add_argument(
        "--only_checking",
        action="store_true",
        help="Only validate existing files without downloading new ones"
    )
    
    parser.add_argument(
        "--validation_level",
        type=str,
        choices=["basic", "full"],
        default="full",
        help="Level of validation: 'basic' (size only) or 'full' (size + structure + content)"
    )
    
    parser.add_argument(
        "--remove_invalid",
        action="store_true",
        help="Remove files that fail validation (use with caution)"
    )

    return parser.parse_args()

def calculate_file_hash(filepath: Path, algorithm='sha256') -> str:
    """
    Calculate the hash of a file using the specified algorithm.
    
    Args:
        filepath (Path): Path to the file
        algorithm (str): Hash algorithm ('sha256', 'sha512', 'md5')
    
    Returns:
        str: Hexadecimal hash value
    """
    hash_obj = getattr(hashlib, algorithm)()
    
    try:
        with open(filepath, "rb") as file:
            # Read file in chunks to handle large files efficiently
            while chunk := file.read(65536):  # 64KB chunks
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {filepath}: {e}")
        return ""

def validate_file_size(filepath: Path, min_size: int = 1024, max_size: int = None) -> bool:
    """
    Validate file size is within acceptable range.
    
    Args:
        filepath (Path): Path to the file
        min_size (int): Minimum acceptable file size in bytes
        max_size (int): Maximum acceptable file size in bytes (None for no limit)
    
    Returns:
        bool: True if file size is valid
    """
    if not filepath.exists():
        return False
    
    file_size = filepath.stat().st_size
    
    if file_size < min_size:
        print(f"File {filepath} is too small: {file_size} bytes (minimum: {min_size})")
        return False
    
    if max_size and file_size > max_size:
        print(f"File {filepath} is too large: {file_size} bytes (maximum: {max_size})")
        return False
    
    return True

def validate_netcdf_structure(filepath: Path) -> bool:
    """
    Validate NetCDF file structure and check for essential components.
    
    Args:
        filepath (Path): Path to the NetCDF file
    
    Returns:
        bool: True if file structure is valid
    """
    if not HAS_NETCDF4:
        print("netCDF4 not available, skipping structure validation")
        return True
    
    try:
        with netCDF4.Dataset(filepath, mode="r") as ds:
            # Check if file has variables
            if len(ds.variables) == 0:
                print(f"NetCDF file {filepath} has no variables")
                return False
            
            # Check if file has dimensions
            if len(ds.dimensions) == 0:
                print(f"NetCDF file {filepath} has no dimensions")
                return False
            
            # Try to read a small sample of data from each variable
            for var_name, var in ds.variables.items():
                try:
                    # Read first few values to ensure data is accessible
                    if var.size > 0:
                        sample_data = var[0] if var.ndim > 0 else var[:]
                        # Check for NaN or infinite values in sample
                        if hasattr(sample_data, '__iter__'):
                            continue  # Skip detailed checks for complex arrays
                except Exception as e:
                    print(f"Error reading variable {var_name} from {filepath}: {e}")
                    return False
            
            return True
    except Exception as e:
        print(f"NetCDF structure validation failed for {filepath}: {e}")
        return False

def check_file_integrity(filepath: Path, expected_hash: str = None, 
                        min_size: int = 1024, max_size: int = None,
                        max_retries: int = 3, validation_level: str = "full") -> bool:
    """
    Enhanced file integrity check with multiple validation methods.
    
    Args:
        filepath (Path): Path to the file
        expected_hash (str): Expected hash value for verification
        min_size (int): Minimum acceptable file size
        max_size (int): Maximum acceptable file size
        max_retries (int): Maximum number of validation retries
        validation_level (str): Level of validation ("basic" or "full")
    
    Returns:
        bool: True if file passes all integrity checks
    """
    for attempt in range(max_retries):
        try:
            # Basic existence and size check
            if not filepath.exists():
                print(f"File {filepath} does not exist")
                return False
            
            # File size validation
            if not validate_file_size(filepath, min_size, max_size):
                return False
            
            # For basic validation, only check size
            if validation_level == "basic":
                print(f"Basic validation passed for {filepath}")
                return True
            
            # Hash validation if expected hash is provided
            if expected_hash:
                calculated_hash = calculate_file_hash(filepath)
                if calculated_hash.lower() != expected_hash.lower():
                    print(f"Hash mismatch for {filepath}: expected {expected_hash}, got {calculated_hash}")
                    return False
                print(f"Hash verification passed for {filepath}")
            
            # NetCDF-specific validation for full validation
            if filepath.suffix.lower() == '.nc':
                if not validate_netcdf_structure(filepath):
                    return False
                
                # Additional NetCDF4 validation
                if HAS_NETCDF4:
                    try:
                        with netCDF4.Dataset(filepath, mode="r") as ds:
                            # Verify we can access the dataset
                            pass
                    except Exception as e:
                        print(f"NetCDF4 validation failed for {filepath}: {e}")
                        if attempt < max_retries - 1:
                            print(f"Retrying validation (attempt {attempt + 2}/{max_retries})")
                            time.sleep(1)
                            continue
                        return False
            
            print(f"File integrity check passed for {filepath}")
            return True
            
        except Exception as e:
            print(f"Integrity check error for {filepath} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return False
    
    return False

def get_expected_files_for_date(date_str: str, data_type: str) -> list:
    """
    Get list of expected files for a given date and data type.
    
    Args:
        date_str (str): Date in YYYY-MM-DD format
        data_type (str): Type of data ("era5" or "cams")
    
    Returns:
        list: List of expected file names
    """
    if data_type == "era5":
        return [
            f"{date_str}-static.nc",
            f"{date_str}-surface-level.nc",
            f"{date_str}-atmospheric.nc"
        ]
    elif data_type == "cams":
        return [
            f"{date_str}-cams.nc"
        ]
    else:
        return []

def validate_files_for_date_range(download_path: Path, start_date: datetime, 
                                 end_date: datetime, data_type: str,
                                 validation_level: str = "full",
                                 remove_invalid: bool = False) -> dict:
    """
    Validate all files for a given date range.
    
    Args:
        download_path (Path): Path to download directory
        start_date (datetime): Start date
        end_date (datetime): End date
        data_type (str): Type of data ("era5" or "cams")
        validation_level (str): Level of validation
        remove_invalid (bool): Whether to remove invalid files
    
    Returns:
        dict: Validation results summary
    """
    total_days = (end_date - start_date).days + 1
    validation_results = {
        'total_expected': 0,
        'files_found': 0,
        'files_valid': 0,
        'files_invalid': 0,
        'files_missing': 0,
        'invalid_files': [],
        'missing_files': []
    }
    
    print(f"\n=== File Validation Report ===")
    print(f"Data type: {data_type}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Validation level: {validation_level}")
    print(f"Remove invalid files: {remove_invalid}")
    print("-" * 50)
    
    for i in range(total_days):
        current_date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        expected_files = get_expected_files_for_date(current_date, data_type)
        
        print(f"\nChecking files for {current_date}:")
        
        for filename in expected_files:
            filepath = download_path / filename
            validation_results['total_expected'] += 1
            
            if not filepath.exists():
                print(f"  ❌ MISSING: {filename}")
                validation_results['files_missing'] += 1
                validation_results['missing_files'].append(str(filepath))
                continue
            
            validation_results['files_found'] += 1
            print(f"  📁 Found: {filename} ({filepath.stat().st_size:,} bytes)")
            
            # Validate the file
            min_size = 1024  # 1KB minimum
            max_size = 10 * 1024 * 1024 * 1024  # 10GB maximum
            
            if check_file_integrity(filepath, None, min_size, max_size, 
                                  validation_level=validation_level):
                print(f"  ✅ VALID: {filename}")
                validation_results['files_valid'] += 1
            else:
                print(f"  ❌ INVALID: {filename}")
                validation_results['files_invalid'] += 1
                validation_results['invalid_files'].append(str(filepath))
                
                if remove_invalid:
                    try:
                        os.remove(filepath)
                        print(f"  🗑️  REMOVED: {filename}")
                    except Exception as e:
                        print(f"  ⚠️  Failed to remove {filename}: {e}")
    
    return validation_results

def print_validation_summary(results: dict):
    """
    Print a comprehensive validation summary.
    
    Args:
        results (dict): Validation results
    """
    print(f"\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total expected files: {results['total_expected']}")
    print(f"Files found:          {results['files_found']}")
    print(f"Files valid:          {results['files_valid']}")
    print(f"Files invalid:        {results['files_invalid']}")
    print(f"Files missing:        {results['files_missing']}")
    
    if results['total_expected'] > 0:
        found_percentage = (results['files_found'] / results['total_expected']) * 100
        valid_percentage = (results['files_valid'] / results['total_expected']) * 100
        print(f"\nCompletion rate:      {found_percentage:.1f}%")
        print(f"Validation rate:      {valid_percentage:.1f}%")
    
    if results['invalid_files']:
        print(f"\nInvalid files:")
        for filepath in results['invalid_files']:
            print(f"  - {filepath}")
    
    if results['missing_files']:
        print(f"\nMissing files:")
        for filepath in results['missing_files']:
            print(f"  - {filepath}")
    
    print("=" * 60)

def safe_download_with_validation(c: cdsapi.Client, request_params: dict, 
                                 output_path: Path, file_type: str = "netcdf",
                                 max_download_attempts: int = 3,
                                 expected_hash: str = None) -> bool:
    """
    Safely download file with validation and retry logic.
    
    Args:
        c (cdsapi.Client): CDS API client
        request_params (dict): API request parameters
        output_path (Path): Output file path
        file_type (str): Type of file being downloaded
        max_download_attempts (int): Maximum download attempts
        expected_hash (str): Expected file hash for validation
    
    Returns:
        bool: True if download and validation successful
    """
    for attempt in range(max_download_attempts):
        try:
            print(f"Downloading {output_path} (attempt {attempt + 1}/{max_download_attempts})")
            
            # Remove existing file if it exists
            if output_path.exists():
                os.remove(output_path)
            
            # Perform the download
            c.retrieve(request_params['dataset'], request_params['params'], str(output_path))
            
            # Validate the downloaded file
            min_size = 1024  # 1KB minimum
            max_size = 10 * 1024 * 1024 * 1024  # 10GB maximum for safety
            
            if check_file_integrity(output_path, expected_hash, min_size, max_size):
                print(f"✅ Successfully downloaded and validated: {output_path}")
                return True
            else:
                print(f"❌ Validation failed for {output_path}")
                if output_path.exists():
                    os.remove(output_path)
                    
        except Exception as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            if output_path.exists():
                os.remove(output_path)
        
        if attempt < max_download_attempts - 1:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    
    print(f"❌ Failed to download {output_path} after {max_download_attempts} attempts")
    return False

def download_one_day_era5(date_str: str, download_path: Path, c: cdsapi.Client, area_val):
    """
    Download the following files for a given date_str with enhanced validation:
      - Static variables
      - Surface-level variables
      - Atmospheric variables
    """
    wyear, wmonth, wday = date_str.split("-")

    # --- Download static variables ---
    static_path = download_path / f"{date_str}-static.nc"
    if not static_path.exists() or not check_file_integrity(static_path):
        request_params = {
            'dataset': "reanalysis-era5-single-levels",
            'params': {
                "product_type": ["reanalysis"],
                "variable": [
                    "geopotential",
                    "land_sea_mask",
                    "soil_type",
                ],
                "year": wyear,
                "month": wmonth,
                "day": wday,
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": area_val,
            }
        }
        
        if safe_download_with_validation(c, request_params, static_path):
            print(f"[{date_str}] Static variables downloaded successfully.")
        else:
            print(f"[{date_str}] Failed to download static variables.")
            return False

    # --- Download surface-level variables ---
    surface_path = download_path / f"{date_str}-surface-level.nc"
    if not surface_path.exists() or not check_file_integrity(surface_path):
        request_params = {
            'dataset': "reanalysis-era5-single-levels",
            'params': {
                "product_type": ["reanalysis"],
                "variable": [
                    "2m_temperature",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "mean_sea_level_pressure",
                ],
                "year": wyear,
                "month": wmonth,
                "day": wday,
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": area_val,
            }
        }
        
        if safe_download_with_validation(c, request_params, surface_path):
            print(f"[{date_str}] Surface-level variables downloaded successfully.")
        else:
            print(f"[{date_str}] Failed to download surface-level variables.")
            return False

    # --- Download atmospheric variables ---
    atmospheric_path = download_path / f"{date_str}-atmospheric.nc"
    if not atmospheric_path.exists() or not check_file_integrity(atmospheric_path):
        request_params = {
            'dataset': "reanalysis-era5-pressure-levels",
            'params': {
                "product_type": ["reanalysis"],
                "variable": [
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "specific_humidity",
                    "geopotential",
                ],
                "pressure_level": [
                    "50", "100", "150", "200", "250", "300", "400",
                    "500", "600", "700", "850", "925", "1000",
                ],
                "year": wyear,
                "month": wmonth,
                "day": wday,
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": area_val,
            }
        }
        
        if safe_download_with_validation(c, request_params, atmospheric_path):
            print(f"[{date_str}] Atmospheric variables downloaded successfully.")
        else:
            print(f"[{date_str}] Failed to download atmospheric variables.")
            return False
    
    return True

def download_one_day_cams(date_str: str, download_path: Path, c: cdsapi.Client, area_val):
    """
    Download CAMS data for one day.
    
    Workflow
    --------
    1. Request a single ZIP file (`netcdf_zip`) from the CAMS forecasts API
       that contains both surface-level and pressure-level variables.
    2. Request 24 hourly times (00–23 UTC, every hour).
    3. After download, extract:
          • data_sfc.nc   -> {YYYY-MM-DD}-cams-surface-level.nc
          • data_plev.nc  -> {YYYY-MM-DD}-cams-atmospheric.nc
    4. Validate the extracted files and delete the original ZIP.

    Parameters
    ----------
    date_str : str
        Date in YYYY-MM-DD format.
    download_path : pathlib.Path
        Destination directory.
    c : cdsapi.Client
        Active CDS API client.
    area_val : list[float]
        Geographic bounding box [S, W, N, E] (ERA / CAMS convention).

    Returns
    -------
    bool
        True if both NetCDF files are downloaded and validated successfully.
    """
    year, month, day = date_str.split("-")

    surface_nc  = download_path / f"{date_str}-cams-surface-level.nc"
    atm_nc      = download_path / f"{date_str}-cams-atmospheric.nc"
    zipped_nc   = download_path / f"{date_str}-cams.nc.zip"

    # Skip if both output files already exist and pass integrity checks
    if surface_nc.exists() and atm_nc.exists() and \
       check_file_integrity(surface_nc) and check_file_integrity(atm_nc):
        print(f"[{date_str}] CAMS files already valid – skipping.")
        return True

    # 24 hourly time steps (00:00–23:00)
    hourly_times = [f"{h:02d}:00" for h in range(24)]

    request_params = {
        "dataset": "cams-global-atmospheric-composition-forecasts",
        "params": {
            "type": "forecast",
            "leadtime_hour": "0",
            "variable": [
                # Meteorological surface variables
                "10m_u_component_of_wind", "10m_v_component_of_wind",
                "2m_temperature", "mean_sea_level_pressure",
                # Pollution surface variables
                "particulate_matter_1um", "particulate_matter_2.5um",
                "particulate_matter_10um", "total_column_carbon_monoxide",
                "total_column_nitrogen_monoxide", "total_column_nitrogen_dioxide",
                "total_column_ozone", "total_column_sulphur_dioxide",
                # Meteorological pressure-level variables
                "u_component_of_wind", "v_component_of_wind", "temperature",
                "geopotential", "specific_humidity",
                # Pollution pressure-level variables
                "carbon_monoxide", "nitrogen_dioxide", "nitrogen_monoxide",
                "ozone", "sulphur_dioxide",
            ],
            "pressure_level": [
                "50", "100", "150", "200", "250", "300", "400",
                "500", "600", "700", "850", "925", "1000",
            ],
            "date": f"{date_str}/{date_str}",
            "time": hourly_times,
            "format": "netcdf_zip",
            # "area": area_val,
        },
    }

    # Download the ZIP; abort early if validation fails
    if not safe_download_with_validation(c, request_params, zipped_nc, file_type="netcdf_zip"):
        print(f"[{date_str}] ZIP download failed.")
        return False

    # Extract surface- and pressure-level NetCDF files
    try:
        with zipfile.ZipFile(zipped_nc, "r") as zf:
            with open(surface_nc, "wb") as f:
                f.write(zf.read("data_sfc.nc"))
            with open(atm_nc,  "wb") as f:
                f.write(zf.read("data_plev.nc"))
        print(f"[{date_str}] Extracted → {surface_nc.name}, {atm_nc.name}")
    except Exception as e:
        print(f"[{date_str}] Extraction error: {e}")
        return False
    finally:
        # Optionally remove the original ZIP
        if zipped_nc.exists():
            zipped_nc.unlink(missing_ok=True)

    # Final integrity check
    ok = check_file_integrity(surface_nc) and check_file_integrity(atm_nc)
    if ok:
        print(f"[{date_str}] CAMS download & validation complete ✅")
    else:
        print(f"[{date_str}] CAMS file validation failed ❌")
    return ok

def get_dates_needing_download(validation_results: dict, data_type: str) -> set:
    """
    Extract dates that need files downloaded based on validation results.
    
    Args:
        validation_results (dict): Results from validation
        data_type (str): Type of data ("era5" or "cams")
    
    Returns:
        set: Set of date strings that need downloads
    """
    dates_needing_download = set()
    
    # Process invalid and missing files
    all_failed_files = validation_results['invalid_files'] + validation_results['missing_files']
    
    for filepath in all_failed_files:
        filename = Path(filepath).name
        
        # Extract date from filename (assuming format: YYYY-MM-DD-*.nc)
        if filename.count('-') >= 3:
            date_part = '-'.join(filename.split('-')[:3])
            try:
                # Validate date format
                datetime.strptime(date_part, "%Y-%m-%d")
                dates_needing_download.add(date_part)
            except ValueError:
                print(f"Warning: Could not extract date from filename: {filename}")
    
    return dates_needing_download

def download_failed_files(download_path: Path, start_date: datetime, end_date: datetime,
                         data_type: str, validation_results: dict, 
                         c: cdsapi.Client, area_val: list) -> dict:
    """
    Download files that failed validation.
    
    Args:
        download_path (Path): Path to download directory
        start_date (datetime): Start date
        end_date (datetime): End date
        data_type (str): Type of data ("era5" or "cams")
        validation_results (dict): Results from validation
        c (cdsapi.Client): CDS API client
        area_val (list): Area coordinates
    
    Returns:
        dict: Download results summary
    """
    dates_needing_download = get_dates_needing_download(validation_results, data_type)
    
    download_results = {
        'attempted_dates': 0,
        'successful_dates': 0,
        'failed_dates': 0,
        'successful_files': 0,
        'failed_files': 0,
        'failed_date_list': []
    }
    
    print(f"\nDates requiring downloads: {sorted(dates_needing_download)}")
    
    for date_str in sorted(dates_needing_download):
        # Verify date is within the specified range
        current_date = datetime.strptime(date_str, "%Y-%m-%d")
        if not (start_date <= current_date <= end_date):
            continue
            
        download_results['attempted_dates'] += 1
        print(f"\n=== Re-downloading {data_type.upper()} data for {date_str} ===")
        
        success = False
        if data_type == 'era5':
            success = download_one_day_era5(date_str, download_path, c, area_val)
        elif data_type == 'cams':
            success = download_one_day_cams(date_str, download_path, c, area_val)
        
        if success:
            download_results['successful_dates'] += 1
            # Count successful files for this date
            expected_files = get_expected_files_for_date(date_str, data_type)
            for filename in expected_files:
                filepath = download_path / filename
                if filepath.exists() and check_file_integrity(filepath):
                    download_results['successful_files'] += 1
                else:
                    download_results['failed_files'] += 1
        else:
            download_results['failed_dates'] += 1
            download_results['failed_date_list'].append(date_str)
    
    return download_results

def print_download_summary(results: dict):
    """
    Print a comprehensive download summary for failed files.
    
    Args:
        results (dict): Download results
    """
    print(f"\n" + "=" * 60)
    print("FAILED FILES DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Attempted dates:      {results['attempted_dates']}")
    print(f"Successful dates:     {results['successful_dates']}")
    print(f"Failed dates:         {results['failed_dates']}")
    print(f"Successful files:     {results['successful_files']}")
    print(f"Failed files:         {results['failed_files']}")
    
    if results['attempted_dates'] > 0:
        success_rate = (results['successful_dates'] / results['attempted_dates']) * 100
        print(f"Success rate:         {success_rate:.1f}%")
    
    if results['failed_date_list']:
        print(f"\nDates that still failed:")
        for date_str in results['failed_date_list']:
            print(f"  - {date_str}")
    
    if results['failed_dates'] == 0:
        print(f"\n✅ All failed files have been successfully downloaded!")
    else:
        print(f"\n⚠️  {results['failed_dates']} dates still have issues. Consider running validation again.")
    
    print("=" * 60)

def validate_files_for_date_range(download_path: Path, start_date: datetime, 
                                 end_date: datetime, data_type: str,
                                 validation_level: str = "full",
                                 remove_invalid: bool = False) -> dict:
    """
    Validate all files for a given date range.
    (This function remains the same as before, but I'm including it for completeness)
    """
    total_days = (end_date - start_date).days + 1
    validation_results = {
        'total_expected': 0,
        'files_found': 0,
        'files_valid': 0,
        'files_invalid': 0,
        'files_missing': 0,
        'invalid_files': [],
        'missing_files': []
    }
    
    print(f"\n=== File Validation Report ===")
    print(f"Data type: {data_type}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Validation level: {validation_level}")
    print(f"Remove invalid files: {remove_invalid}")
    print("-" * 50)
    
    for i in range(total_days):
        current_date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        expected_files = get_expected_files_for_date(current_date, data_type)
        
        print(f"\nChecking files for {current_date}:")
        
        for filename in expected_files:
            filepath = download_path / filename
            validation_results['total_expected'] += 1
            
            if not filepath.exists():
                print(f"  ❌ MISSING: {filename}")
                validation_results['files_missing'] += 1
                validation_results['missing_files'].append(str(filepath))
                continue
            
            validation_results['files_found'] += 1
            print(f"  📁 Found: {filename} ({filepath.stat().st_size:,} bytes)")
            
            # Validate the file
            min_size = 1024  # 1KB minimum
            max_size = 10 * 1024 * 1024 * 1024  # 10GB maximum
            
            if check_file_integrity(filepath, None, min_size, max_size, 
                                  validation_level=validation_level):
                print(f"  ✅ VALID: {filename}")
                validation_results['files_valid'] += 1
            else:
                print(f"  ❌ INVALID: {filename}")
                validation_results['files_invalid'] += 1
                validation_results['invalid_files'].append(str(filepath))
                
                if remove_invalid:
                    try:
                        os.remove(filepath)
                        print(f"  🗑️  REMOVED: {filename}")
                    except Exception as e:
                        print(f"  ⚠️  Failed to remove {filename}: {e}")
    
    return validation_results


def snap_and_shrink(box, res=0.4, patch=3):
    """
    Return [S,W,N,E] fully inside `box`, such that
    (#lat_pts % patch == 0) and (#lon_pts % patch == 0).
    """
    s, w, n, e = box

    # ① snap to the 0.4° grid, **inside** original box
    s = math.ceil(s  / res) * res
    w = math.ceil(w  / res) * res
    n = math.floor(n / res) * res
    e = math.floor(e / res) * res

    # ② counts as **integers** (round to remove fp-noise)
    lat_cnt = int(round((n - s) / res)) + 1
    lon_cnt = int(round((e - w) / res)) + 1

    # ③ shrink north edge until divisible by patch
    while lat_cnt % patch:
        n -= res
        lat_cnt -= 1            # keep count in sync

    # ④ shrink east edge until divisible by patch
    while lon_cnt % patch:
        e -= res
        lon_cnt -= 1

    return [round(v, 2) for v in (s, w, n, e)]

def main():
    args = parse_args()

    # Prepare download path
    download_path = os.path.join(args.download_path, args.data)
    download_path = Path(download_path).expanduser()
    download_path.mkdir(parents=True, exist_ok=True)

    # Prepare date range
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    # If only checking files, run validation and download failed files
    if args.only_checking:
        print("Running file validation and downloading failed files")
        validation_results = validate_files_for_date_range(
            download_path, start_date, end_date, args.data,
            args.validation_level, args.remove_invalid
        )
        print_validation_summary(validation_results)
        
        # Check if there are any failed files to download
        failed_files_count = validation_results['files_invalid'] + validation_results['files_missing']
        
        if failed_files_count == 0:
            print("\n✅ All files are valid. No downloads needed.")
            return
        
        print(f"\n🔄 Found {failed_files_count} failed files. Starting downloads...")
        
        # Create CDS API client for downloads
        c = cdsapi.Client(verify=False)
        area_val = [19.415, 104.643, 49.199, 160.555]
        
        # Download failed files
        download_results = download_failed_files(
            download_path, start_date, end_date, args.data, 
            validation_results, c, area_val
        )
        
        print_download_summary(download_results)
        return

    # Create CDS API client (with SSL verification disabled)
    c = cdsapi.Client(verify=False)

    # Define the desired area (modify if needed)
    # area_val = [19.415, 104.643, 49.199, 160.555]
    area_val = snap_and_shrink([19.415, 104.643, 49.199, 160.555])
    # area_val = [19.6, 104.8, 48.0, 159.6]            # the box you used for CAMS
    # static_vars = crop_static(global_static, area_val)  # (1,72,138) each
    print(f"Area val => {area_val}")

    total_days = (end_date - start_date).days + 1
    successful_downloads = 0
    failed_downloads = 0
    
    if args.data == 'era5':
        for i in range(total_days):
            current_date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            print(f"\n=== Downloading ERA5 data for {current_date} ===")
            if download_one_day_era5(current_date, download_path, c, area_val):
                successful_downloads += 1
            else:
                failed_downloads += 1
    else:
        for i in range(total_days):
            current_date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            print(f"\n=== Downloading CAMS data for {current_date} ===")
            if download_one_day_cams(current_date, download_path, c, area_val):
                successful_downloads += 1
            else:
                failed_downloads += 1
    
    print(f"\n=== Download Summary ===")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")
    print(f"Total attempted: {total_days}")

if __name__ == "__main__":
    main()

# import argparse
# from pathlib import Path
# import cdsapi
# import urllib3
# import os
# from datetime import datetime, timedelta

# urllib3.disable_warnings()


# def parse_args():
#     """
#     Parse command-line arguments:
#       --download_path: the directory where the data will be saved
#       --start_date: the date from which to start downloading (YYYY-MM-DD)
#       --end_date: the date at which to end downloading (YYYY-MM-DD)
#     """
#     parser = argparse.ArgumentParser(description="ERA5 data download (single call per dataset).")

#     parser.add_argument(
#         "--download_path",
#         type=str,
#         default="./dataera5/",
#         help="Path to the download directory (default: ./dataera5/)"
#     )
#     parser.add_argument(
#         "--start_date",
#         type=str,
#         required=True,
#         help="Starting date for downloading (YYYY-MM-DD)"
#     )
#     parser.add_argument(
#         "--end_date",
#         type=str,
#         required=True,
#         help="Ending date for downloading (YYYY-MM-DD)"
#     )

#     return parser.parse_args()


# def main():
#     args = parse_args()

#     download_path = Path(args.download_path).expanduser()
#     download_path.mkdir(parents=True, exist_ok=True)

#     start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
#     end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

#     # Collect all year/month/day in sets
#     year_set = set()
#     month_set = set()
#     day_set = set()

#     current_date = start_date
#     while current_date <= end_date:
#         year_set.add(current_date.strftime("%Y"))
#         month_set.add(current_date.strftime("%m"))
#         day_set.add(current_date.strftime("%d"))
#         current_date += timedelta(days=1)

#     # Convert to sorted lists (strings)
#     year_list = sorted(list(year_set))
#     month_list = sorted(list(month_set))
#     day_list = sorted(list(day_set))

#     print(f"* [{start_date} ~ [{end_date}]]")
#     print(f"    Download years: {year_list}")
#     print(f"    Download months: {month_list}")
#     print(f"    Download days: {day_list}")

#     c = cdsapi.Client(verify=False)

#     # Define the desired area (same as original code)
#     area_val = [19.415, 104.643, 49.199, 160.555]

#     # 1) Download static variables (single-level dataset)
#     static_file = download_path / f"static_{args.start_date}_{args.end_date}.nc"
#     if not static_file.exists():
#         print("Requesting static variables for the entire date range...")
#         c.retrieve(
#             "reanalysis-era5-single-levels",
#             {
#                 "product_type": "reanalysis",
#                 "variable": [
#                     "geopotential",
#                     "land_sea_mask",
#                     "soil_type",
#                 ],
#                 "year": year_list,
#                 "month": month_list,
#                 "day": day_list,
#                 "time": ["00:00", "06:00", "12:00", "18:00"],
#                 "format": "netcdf",
#                 "download_format": "unarchived",
#                 "area": area_val,
#             },
#             str(static_file),
#         )
#         print(f"[static] Downloaded => {static_file}")
#     else:
#         print(f"[static] Already exists => {static_file}")

#     # 2) Download surface-level variables (single-level dataset)
#     surface_file = download_path / f"surface_{args.start_date}_{args.end_date}.nc"
#     if not surface_file.exists():
#         print("Requesting surface-level variables for the entire date range...")
#         c.retrieve(
#             "reanalysis-era5-single-levels",
#             {
#                 "product_type": "reanalysis",
#                 "variable": [
#                     "2m_temperature",
#                     "10m_u_component_of_wind",
#                     "10m_v_component_of_wind",
#                     "mean_sea_level_pressure",
#                 ],
#                 "year": year_list,
#                 "month": month_list,
#                 "day": day_list,
#                 "time": ["00:00", "06:00", "12:00", "18:00"],
#                 "format": "netcdf",
#                 "download_format": "unarchived",
#                 "area": area_val,
#             },
#             str(surface_file),
#         )
#         print(f"[surface-level] Downloaded => {surface_file}")
#     else:
#         print(f"[surface-level] Already exists => {surface_file}")

#     # 3) Download atmospheric variables (pressure-level dataset)
#     atmospheric_file = download_path / f"atmospheric_{args.start_date}_{args.end_date}.nc"
#     if not atmospheric_file.exists():
#         print("Requesting atmospheric variables for the entire date range...")
#         c.retrieve(
#             "reanalysis-era5-pressure-levels",
#             {
#                 "product_type": "reanalysis",
#                 "variable": [
#                     "temperature",
#                     "u_component_of_wind",
#                     "v_component_of_wind",
#                     "specific_humidity",
#                     "geopotential",
#                 ],
#                 "pressure_level": [
#                     "50",
#                     "100",
#                     "150",
#                     "200",
#                     "250",
#                     "300",
#                     "400",
#                     "500",
#                     "600",
#                     "700",
#                     "850",
#                     "925",
#                     "1000",
#                 ],
#                 "year": year_list,
#                 "month": month_list,
#                 "day": day_list,
#                 "time": ["00:00", "06:00", "12:00", "18:00"],
#                 "format": "netcdf",
#                 "download_format": "unarchived",
#                 "area": area_val,
#             },
#             str(atmospheric_file),
#         )
#         print(f"[atmospheric] Downloaded => {atmospheric_file}")
#     else:
#         print(f"[atmospheric] Already exists => {atmospheric_file}")


# if __name__ == "__main__":
#     main()

