"""
Simple test for code quality improvement system
"""

import os
import sys
import asyncio
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.getcwd())

async def test_code_quality():
    print("Testing code quality improvement system...")

    try:
        from code_quality_improvement_system import CodeQualityAnalyzer, CodeQualityImprover

        # Mock dependencies
        class MockAuditLogger:
            pass

        class MockCommunication:
            pass

        analyzer = CodeQualityAnalyzer(MockAuditLogger(), MockCommunication())
        improver = CodeQualityImprover(analyzer)

        print("✅ Imports successful")

        # Run basic analysis
        results = await analyzer.run_full_quality_analysis()
        print(f"✅ Analysis completed: {results['files_analyzed']} files analyzed")
        print(f"   Quality score: {results['quality_score']}%")
        print(f"   Issues found: {results['issues_found']}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_code_quality())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")