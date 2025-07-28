#!/usr/bin/env python3
"""
Quick Start Script for Email-SMS Intelligent Model
Get up and running in minutes!
"""

import os
import sys
import subprocess
import time

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🚀 Email-SMS Intelligent Model - Quick Start")
    print("=" * 60)
    print("A comprehensive solution for financial email/SMS classification")
    print("and data extraction using transformer models and MongoDB.")
    print("=" * 60)

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def download_spacy_model():
    """Download spaCy model"""
    print("\n📥 Downloading spaCy model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download spaCy model: {e}")
        return False

def setup_environment():
    """Setup environment variables"""
    print("\n🔧 Setting up environment...")
    
    # Create .env file if it doesn't exist
    env_file = ".env"
    if not os.path.exists(env_file):
        with open(env_file, "w") as f:
            f.write("MONGODB_URI=mongodb://localhost:27017\n")
            f.write("DATABASE_NAME=financial_assistant\n")
        print("✅ Created .env file with default MongoDB settings")
    else:
        print("✅ .env file already exists")
    
    # Create necessary directories
    directories = ["models", "logs", "cache", "results"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}/")

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    try:
        subprocess.check_call([sys.executable, "test_model.py"])
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Some tests failed: {e}")
        return False

def run_demo():
    """Run a quick demo"""
    print("\n🎯 Running quick demo...")
    
    try:
        # Import our modules
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from model import EmailSMSModel
        
        # Initialize model
        print("🤖 Initializing model...")
        model = EmailSMSModel()
        
        # Test cases
        test_cases = [
            "Your HDFC Bank account has been debited with Rs. 500 for UPI transaction to Zomato",
            "Hello, how are you doing today?",
            "Dear Customer, Rs.1000.00 has been credited to your account by VPA john@okaxis",
            "Your Netflix subscription has been renewed for Rs. 499"
        ]
        
        print("\n📊 Demo Results:")
        print("-" * 50)
        
        for i, text in enumerate(test_cases, 1):
            result = model.predict(text)
            status = "💰 Financial" if result['is_financial'] else "📧 Non-Financial"
            print(f"{i}. {status} (Confidence: {result['financial_confidence']:.3f})")
            print(f"   Text: {text[:60]}...")
            
            if result['extraction'] and result['extraction']['amount']:
                print(f"   💳 Amount: Rs. {result['extraction']['amount']}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def show_next_steps():
    """Show next steps"""
    print("\n" + "=" * 60)
    print("🎉 Quick Start Complete!")
    print("=" * 60)
    print("\n📋 Next Steps:")
    print("1. 🚀 Train the model: python train.py")
    print("2. 📊 Use the notebook: jupyter notebook training_notebook.ipynb")
    print("3. 🗄️ Set up MongoDB (optional): mongod")
    print("4. 🔧 Customize config: Edit config.py")
    print("5. 📚 Read documentation: README.md")
    
    print("\n🔗 Useful Commands:")
    print("• python train.py --help                    # Training options")
    print("• python test_model.py                      # Run tests")
    print("• python mongodb_integration.py             # Test MongoDB")
    print("• python data_preprocessing.py              # Process datasets")
    
    print("\n📁 Project Structure:")
    print("• config.py                 # Configuration")
    print("• model.py                  # Main model")
    print("• train.py                  # Training script")
    print("• mongodb_integration.py    # Database operations")
    print("• training_notebook.ipynb   # Interactive notebook")
    
    print("\n🎯 Integration with Agno:")
    print("• Use mongodb_integration.py for data storage")
    print("• Use model.py for predictions")
    print("• Use train.py for model training")
    print("• Check README.md for API examples")
    
    print("\n" + "=" * 60)
    print("Happy coding! 🚀")
    print("=" * 60)

def main():
    """Main quick start function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Download spaCy model
    if not download_spacy_model():
        return False
    
    # Setup environment
    setup_environment()
    
    # Test installation
    if not test_installation():
        print("⚠️ Installation test failed, but continuing...")
    
    # Run demo
    if not run_demo():
        print("⚠️ Demo failed, but installation may still be working...")
    
    # Show next steps
    show_next_steps()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Quick start completed successfully!")
        else:
            print("\n❌ Quick start encountered issues. Please check the errors above.")
    except KeyboardInterrupt:
        print("\n\n⚠️ Quick start interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the installation and try again.") 