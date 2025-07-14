"""
Azure Blob Storage Connection Test Script

This script tests the connection to Azure Blob Storage and lists files/folders
in the specified container and path.

Make sure to configure your .env file with the appropriate credentials.
"""

import os
import sys
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import AzureError, ResourceNotFoundError
from dotenv import load_dotenv

def load_config():
    """Load configuration from environment variables"""
    load_dotenv()
    
    config = {
        'account_name': os.getenv('AZURE_STORAGE_ACCOUNT_NAME'),
        'container_name': os.getenv('AZURE_STORAGE_CONTAINER_NAME'),
        'connection_string': os.getenv('AZURE_STORAGE_CONNECTION_STRING'),
        'sas_token': os.getenv('AZURE_STORAGE_SAS_TOKEN'),
        'account_key': os.getenv('AZURE_STORAGE_ACCOUNT_KEY'),
        'folder_path': os.getenv('AZURE_BLOB_FOLDER_PATH', ''),
        'filter_pdf_files': os.getenv('FILTER_PDF_FILES', 'false').lower() == 'true',
        'show_file_summary': os.getenv('SHOW_FILE_SUMMARY', 'false').lower() == 'true'
    }
    
    return config

def create_blob_service_client(config):
    """Create BlobServiceClient using available authentication method (in priority order)"""
    
    # Method 1: Connection String (PRIMARY - Most reliable)
    if config['connection_string'] and config['connection_string'].strip():
        print("🔑 Using Connection String authentication (PRIMARY)")
        try:
            return BlobServiceClient.from_connection_string(config['connection_string'])
        except Exception as e:
            print(f"⚠️ Connection String failed: {str(e)}")
            print("🔄 Falling back to SAS Token...")
    
    # Method 2: SAS Token (FALLBACK 1)
    if config['account_name'] and config['sas_token'] and config['sas_token'].strip():
        print("🔑 Using SAS Token authentication (FALLBACK 1)")
        try:
            account_url = f"https://{config['account_name']}.blob.core.windows.net"
            return BlobServiceClient(account_url=account_url, credential=config['sas_token'])
        except Exception as e:
            print(f"⚠️ SAS Token failed: {str(e)}")
            print("🔄 Falling back to Account Key...")
    
    # Method 3: Account Key (FALLBACK 2)
    if config['account_name'] and config['account_key'] and config['account_key'].strip():
        print("🔑 Using Account Key authentication (FALLBACK 2)")
        account_url = f"https://{config['account_name']}.blob.core.windows.net"
        return BlobServiceClient(account_url=account_url, credential=config['account_key'])
    
    # No valid authentication found
    error_msg = (
        "❌ No valid authentication method configured. Please set one of:\n"
        "PRIMARY: AZURE_STORAGE_CONNECTION_STRING (most reliable)\n"
        "FALLBACK 1: AZURE_STORAGE_SAS_TOKEN\n"
        "FALLBACK 2: AZURE_STORAGE_ACCOUNT_KEY\n\n"
        "📝 Note: Connection string format should be:\n"
        "DefaultEndpointsProtocol=https;AccountName=edifystorageaccount;AccountKey=YOUR_KEY;EndpointSuffix=core.windows.net"
    )
    raise ValueError(error_msg)

def test_connection(blob_service_client, container_name):
    """Test the connection by trying to access the container"""
    try:
        container_client = blob_service_client.get_container_client(container_name)
        
        # Test if container exists and is accessible
        container_properties = container_client.get_container_properties()
        print(f"✅ Successfully connected to container: {container_name}")
        print(f"📅 Container last modified: {container_properties.last_modified}")
        return container_client
        
    except ResourceNotFoundError:
        print(f"❌ Container '{container_name}' not found")
        return None
    except AzureError as e:
        print(f"❌ Connection failed: {str(e)}")
        return None

def list_blobs(container_client, folder_path="", config=None):
    """List all blobs in the container with optional folder prefix and PDF filtering"""
    try:
        print(f"\n📂 Listing contents in container")
        if folder_path:
            print(f"🔍 Filtering by folder path: {folder_path}")
        
        filter_pdf = config.get('filter_pdf_files', False) if config else False
        show_summary = config.get('show_file_summary', False) if config else False
        
        if filter_pdf:
            print("📋 Filtering: PDF files only")
        
        # Get list of blobs
        blob_list = container_client.list_blobs(name_starts_with=folder_path)
        
        files = []
        pdf_files = []
        folders = set()
        total_size = 0
        pdf_total_size = 0
        
        for blob in blob_list:
            blob_name = blob.name
            
            # Remove the folder prefix for cleaner display
            display_name = blob_name
            if folder_path and blob_name.startswith(folder_path):
                display_name = blob_name[len(folder_path):]
            
            # Check if this is a "folder" (blob name ends with /)
            if blob_name.endswith('/'):
                folders.add(display_name)
            else:
                # Check if file is in a subfolder
                if '/' in display_name:
                    folder_name = display_name.split('/')[0] + '/'
                    folders.add(folder_name)
                
                file_info = {
                    'name': display_name,
                    'full_name': blob_name,
                    'size': blob.size or 0,
                    'last_modified': blob.last_modified,
                    'is_pdf': display_name.lower().endswith('.pdf')
                }
                
                files.append(file_info)
                total_size += file_info['size']
                
                # Track PDF files separately
                if file_info['is_pdf']:
                    pdf_files.append(file_info)
                    pdf_total_size += file_info['size']
        
        # Filter files if PDF filtering is enabled
        display_files = pdf_files if filter_pdf else files
        
        # Display results
        if filter_pdf:
            print(f"\n📊 Found {len(folders)} folder(s) and {len(pdf_files)} PDF file(s) (out of {len(files)} total files)")
        else:
            print(f"\n📊 Found {len(folders)} folder(s) and {len(files)} file(s)")
            if pdf_files:
                print(f"📄 Including {len(pdf_files)} PDF file(s)")
        
        if folders:
            print("\n📁 Folders:")
            for folder in sorted(folders):
                print(f"  📂 {folder}")
        
        if display_files:
            file_type = "PDF Files" if filter_pdf else "Files"
            print(f"\n📄 {file_type}:")
            for file in sorted(display_files, key=lambda x: x['name']):
                size_mb = file['size'] / (1024 * 1024) if file['size'] else 0
                pdf_icon = "📕" if file['is_pdf'] else "📄"
                print(f"  {pdf_icon} {file['name']} ({size_mb:.2f} MB) - {file['last_modified']}")
        
        # Show summary if enabled
        if show_summary:
            print(f"\n📈 SUMMARY:")
            print(f"  📁 Total Folders: {len(folders)}")
            print(f"  📄 Total Files: {len(files)}")
            print(f"  📕 PDF Files: {len(pdf_files)}")
            print(f"  💾 Total Size: {total_size / (1024 * 1024):.2f} MB")
            if pdf_files:
                print(f"  📕 PDF Total Size: {pdf_total_size / (1024 * 1024):.2f} MB")
                print(f"  📊 PDFs represent {(len(pdf_files) / len(files) * 100):.1f}% of all files")
                print(f"  💽 PDFs represent {(pdf_total_size / total_size * 100):.1f}% of total storage")
        
        if not display_files and not folders:
            print("📭 No files or folders found in the specified path")
            
    except AzureError as e:
        print(f"❌ Error listing blobs: {str(e)}")

def main():
    """Main function to test Azure Blob Storage connection"""
    print("🔄 Azure Blob Storage Connection Test\n")
    
    try:
        # Load configuration
        config = load_config()
        
        # Validate required configuration
        if not config['account_name']:
            print("❌ AZURE_STORAGE_ACCOUNT_NAME is required")
            return
        
        if not config['container_name']:
            print("❌ AZURE_STORAGE_CONTAINER_NAME is required")
            return
        
        print(f"🏢 Account Name: {config['account_name']}")
        print(f"📦 Container Name: {config['container_name']}")
        print(f"📁 Folder Path: {config['folder_path'] or '(root)'}")
        print(f"📋 PDF Filter: {'Enabled' if config['filter_pdf_files'] else 'Disabled'}")
        print(f"📈 Summary: {'Enabled' if config['show_file_summary'] else 'Disabled'}")
        
        # Create blob service client
        blob_service_client = create_blob_service_client(config)
        
        # Test connection
        container_client = test_connection(blob_service_client, config['container_name'])
        
        if container_client:
            # List blobs if connection successful
            list_blobs(container_client, config['folder_path'], config)
        else:
            print("\n❌ Cannot proceed with listing due to connection failure")
            
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return

if __name__ == "__main__":
    main()
