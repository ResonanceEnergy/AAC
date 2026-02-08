#!/usr/bin/env python3
"""
Deep Dive File Analyzer
=======================
Advanced file analysis system for strategy-related data discovery and insights.
Scans system files, extracts relevant metrics, and provides actionable intelligence.
"""

import asyncio
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys
import re
import ast
import inspect
import argparse
from pathlib import Path
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


class DeepDiveFileAnalyzer:
    """
    Comprehensive file analysis system for discovering strategy-related data,
    metrics, and insights across the entire codebase.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analysis_cache: Dict[str, Any] = {}
        self.file_index: Dict[str, Dict] = {}
        self.metric_patterns = self._load_metric_patterns()
        self.strategy_patterns = self._load_strategy_patterns()

    def _load_metric_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for identifying metrics in code"""
        return {
            'performance_metrics': [
                r'sharpe.*ratio', r'total.*return', r'annualized.*return',
                r'win.*rate', r'profit.*factor', r'max.*drawdown'
            ],
            'risk_metrics': [
                r'volatility', r'value.*at.*risk', r'expected.*shortfall',
                r'beta', r'correlation', r'stress.*test'
            ],
            'trading_metrics': [
                r'position.*size', r'stop.*loss', r'take.*profit',
                r'leverage', r'margin', r'commission'
            ],
            'market_data': [
                r'price.*data', r'volume.*data', r'order.*book',
                r'tick.*data', r'historical.*data', r'real.*time.*data'
            ],
            'strategy_parameters': [
                r'entry.*threshold', r'exit.*threshold', r'lookback.*period',
                r'min.*signal', r'max.*position', r'risk.*limit'
            ]
        }

    def _load_strategy_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for identifying strategy components"""
        return {
            'arbitrage_types': [
                r'statistical.*arbitrage', r'triangular.*arbitrage',
                r'cross.*exchange', r'merger.*arbitrage', r'convertible.*arbitrage'
            ],
            'signal_generation': [
                r'signal.*generation', r'entry.*signal', r'exit.*signal',
                r'buy.*signal', r'sell.*signal', r'neutral.*signal'
            ],
            'execution_logic': [
                r'order.*execution', r'position.*management', r'risk.*management',
                r'portfolio.*optimization', r'asset.*allocation'
            ],
            'backtesting': [
                r'backtest', r'historical.*simulation', r'paper.*trading',
                r'performance.*analysis', r'walk.*forward'
            ]
        }

    async def perform_comprehensive_scan(self, root_path: Optional[Path] = None) -> Dict[str, Any]:
        """Perform comprehensive scan of all relevant files"""
        if root_path is None:
            root_path = PROJECT_ROOT

        self.logger.info(f"Starting comprehensive file scan from {root_path}")

        scan_results = {
            'scan_timestamp': datetime.now().isoformat(),
            'total_files': 0,
            'strategy_files': [],
            'metric_files': [],
            'data_files': [],
            'config_files': [],
            'test_files': [],
            'analysis_summary': {},
            'insights': [],
            'recommendations': []
        }

        # Scan all Python files
        python_files = list(root_path.rglob("*.py"))
        scan_results['total_files'] = len(python_files)

        for file_path in python_files:
            if self._should_analyze_file(file_path):
                file_analysis = await self._analyze_file_deep(file_path)
                self._categorize_file(file_analysis, scan_results)

        # Generate analysis summary
        scan_results['analysis_summary'] = self._generate_analysis_summary(scan_results)

        # Generate insights and recommendations
        scan_results['insights'] = self._generate_insights(scan_results)
        scan_results['recommendations'] = self._generate_recommendations(scan_results)

        self.logger.info(f"Comprehensive scan completed: {scan_results['total_files']} files analyzed")
        return scan_results

    def _should_analyze_file(self, file_path: Path) -> bool:
        """Determine if a file should be analyzed"""
        # Skip common exclude patterns
        exclude_patterns = [
            '__pycache__', '.git', 'node_modules', '.venv',
            'build', 'dist', '*.pyc', '.pytest_cache'
        ]

        path_str = str(file_path)
        for pattern in exclude_patterns:
            if pattern in path_str:
                return False

        # Only analyze Python files for now
        return file_path.suffix == '.py'

    async def _analyze_file_deep(self, file_path: Path) -> Dict[str, Any]:
        """Perform deep analysis of a single file"""
        cache_key = str(file_path)
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]

        analysis = {
            'file_path': str(file_path.relative_to(PROJECT_ROOT)),
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            'categories': [],
            'metrics_found': {},
            'strategies_found': {},
            'code_quality': {},
            'dependencies': [],
            'complexity_score': 0,
            'test_coverage': 0,
            'documentation_score': 0
        }

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Analyze content
            analysis['code_quality'] = self._analyze_code_quality(content)
            analysis['metrics_found'] = self._find_metrics_in_content(content)
            analysis['strategies_found'] = self._find_strategies_in_content(content)
            analysis['dependencies'] = self._extract_dependencies(content)
            analysis['complexity_score'] = self._calculate_complexity_score(content)
            analysis['documentation_score'] = self._calculate_documentation_score(content)

            # Categorize file
            analysis['categories'] = self._categorize_file_content(content)

        except Exception as e:
            self.logger.warning(f"Error analyzing {file_path}: {e}")
            analysis['error'] = str(e)

        self.analysis_cache[cache_key] = analysis
        return analysis

    def _analyze_code_quality(self, content: str) -> Dict[str, Any]:
        """Analyze code quality metrics"""
        lines = content.split('\n')
        total_lines = len(lines)
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]

        return {
            'total_lines': total_lines,
            'code_lines': len(code_lines),
            'comment_lines': total_lines - len(code_lines),
            'functions': content.count('def '),
            'classes': content.count('class '),
            'imports': len(re.findall(r'^(from|import)\s', content, re.MULTILINE)),
            'error_handling': content.count('try:'),
            'logging_statements': content.count('logger.'),
            'docstrings': content.count('"""') // 2  # Each docstring has 2 """
        }

    def _find_metrics_in_content(self, content: str) -> Dict[str, List[str]]:
        """Find metrics-related content in file"""
        metrics_found = {}

        for category, patterns in self.metric_patterns.items():
            found_items = []
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    found_items.extend(matches)
            if found_items:
                metrics_found[category] = list(set(found_items))

        return metrics_found

    def _find_strategies_in_content(self, content: str) -> Dict[str, List[str]]:
        """Find strategy-related content in file"""
        strategies_found = {}

        for category, patterns in self.strategy_patterns.items():
            found_items = []
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    found_items.extend(matches)
            if found_items:
                strategies_found[category] = list(set(found_items))

        return strategies_found

    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract import dependencies"""
        dependencies = []

        # Find import statements
        import_pattern = r'^(?:from\s+|\s*import\s+)([\w.]+)'
        matches = re.findall(import_pattern, content, re.MULTILINE)

        for match in matches:
            # Clean up the import
            dep = match.split('.')[0]
            if dep not in ['sys', 'os', 'json', 'datetime', 'typing', 'pathlib'] and dep not in dependencies:
                dependencies.append(dep)

        return dependencies

    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate code complexity score"""
        score = 0

        # Factors contributing to complexity
        score += content.count('if ') * 0.5
        score += content.count('for ') * 0.3
        score += content.count('while ') * 0.3
        score += content.count('try:') * 0.2
        score += content.count('async def ') * 0.1
        score += len(content.split('\n')) * 0.01  # File size factor

        # Nested structures (rough estimate)
        nested_score = content.count('    ') * 0.1
        score += nested_score

        return min(100, score)  # Cap at 100

    def _calculate_documentation_score(self, content: str) -> float:
        """Calculate documentation score"""
        score = 0

        # Docstring presence
        if '"""' in content:
            score += 30

        # Comment ratio
        lines = content.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        comment_ratio = comment_lines / len(lines) if lines else 0
        score += comment_ratio * 40

        # Function documentation
        functions = content.count('def ')
        docstrings = content.count('"""') // 2
        if functions > 0:
            docstring_ratio = docstrings / functions
            score += docstring_ratio * 30

        return min(100, score)

    def _categorize_file_content(self, content: str) -> List[str]:
        """Categorize file based on content analysis"""
        categories = []

        # Strategy files
        if any(pattern in content.lower() for pattern in ['strategy', 'arbitrage', 'trading']):
            categories.append('strategy')

        # Test files
        if any(pattern in content.lower() for pattern in ['test', 'unittest', 'pytest']):
            categories.append('test')

        # Data files
        if any(pattern in content.lower() for pattern in ['data', 'csv', 'json', 'database']):
            categories.append('data')

        # Config files
        if any(pattern in content.lower() for pattern in ['config', 'settings', 'parameters']):
            categories.append('config')

        # Analysis files
        if any(pattern in content.lower() for pattern in ['analysis', 'metrics', 'report']):
            categories.append('analysis')

        return categories

    def _categorize_file(self, file_analysis: Dict, scan_results: Dict):
        """Categorize file into appropriate result categories"""
        categories = file_analysis.get('categories', [])

        if 'strategy' in categories:
            scan_results['strategy_files'].append(file_analysis)
        if 'test' in categories:
            scan_results['test_files'].append(file_analysis)
        if 'data' in categories:
            scan_results['data_files'].append(file_analysis)
        if 'config' in categories:
            scan_results['config_files'].append(file_analysis)

        # Also check for metrics
        if file_analysis.get('metrics_found'):
            scan_results['metric_files'].append(file_analysis)

    def _generate_analysis_summary(self, scan_results: Dict) -> Dict[str, Any]:
        """Generate summary statistics from scan results"""
        summary = {
            'total_files_analyzed': scan_results['total_files'],
            'strategy_files_count': len(scan_results['strategy_files']),
            'test_files_count': len(scan_results['test_files']),
            'data_files_count': len(scan_results['data_files']),
            'config_files_count': len(scan_results['config_files']),
            'metric_files_count': len(scan_results['metric_files']),
            'code_quality_stats': {},
            'complexity_distribution': {},
            'documentation_coverage': {}
        }

        # Aggregate code quality stats
        all_files = (scan_results['strategy_files'] + scan_results['test_files'] +
                    scan_results['data_files'] + scan_results['config_files'])

        if all_files:
            quality_scores = [f.get('code_quality', {}) for f in all_files]
            summary['code_quality_stats'] = {
                'avg_functions_per_file': np.mean([q.get('functions', 0) for q in quality_scores]),
                'avg_classes_per_file': np.mean([q.get('classes', 0) for q in quality_scores]),
                'avg_lines_per_file': np.mean([q.get('total_lines', 0) for q in quality_scores])
            }

            complexity_scores = [f.get('complexity_score', 0) for f in all_files]
            summary['complexity_distribution'] = {
                'low': len([s for s in complexity_scores if s < 20]),
                'medium': len([s for s in complexity_scores if 20 <= s < 50]),
                'high': len([s for s in complexity_scores if s >= 50])
            }

            doc_scores = [f.get('documentation_score', 0) for f in all_files]
            summary['documentation_coverage'] = {
                'well_documented': len([s for s in doc_scores if s > 70]),
                'moderately_documented': len([s for s in doc_scores if 40 <= s <= 70]),
                'poorly_documented': len([s for s in doc_scores if s < 40])
            }

        return summary

    def _generate_insights(self, scan_results: Dict) -> List[str]:
        """Generate insights from scan results"""
        insights = []

        summary = scan_results.get('analysis_summary', {})

        # Strategy coverage insights
        strategy_count = summary.get('strategy_files_count', 0)
        if strategy_count < 10:
            insights.append(f"Low strategy implementation: Only {strategy_count} strategy files found")
        elif strategy_count > 20:
            insights.append(f"Good strategy coverage: {strategy_count} strategy files implemented")

        # Test coverage insights
        test_count = summary.get('test_files_count', 0)
        if test_count == 0:
            insights.append("Critical: No test files found - high risk for production deployment")
        elif test_count < strategy_count * 0.5:
            insights.append(f"Low test coverage: Only {test_count} test files for {strategy_count} strategy files")

        # Code quality insights
        quality_stats = summary.get('code_quality_stats', {})
        avg_lines = quality_stats.get('avg_lines_per_file', 0)
        if avg_lines > 300:
            insights.append(f"Large files detected: Average {avg_lines:.0f} lines per file may indicate complexity issues")

        # Complexity insights
        complexity_dist = summary.get('complexity_distribution', {})
        high_complexity = complexity_dist.get('high', 0)
        if high_complexity > 0:
            insights.append(f"High complexity code: {high_complexity} files have high complexity scores")

        # Documentation insights
        doc_coverage = summary.get('documentation_coverage', {})
        poorly_doc = doc_coverage.get('poorly_documented', 0)
        if poorly_doc > 0:
            insights.append(f"Documentation gap: {poorly_doc} files have poor documentation")

        return insights

    def _generate_recommendations(self, scan_results: Dict) -> List[str]:
        """Generate recommendations based on scan results"""
        recommendations = []

        summary = scan_results.get('analysis_summary', {})

        # Testing recommendations
        test_count = summary.get('test_files_count', 0)
        strategy_count = summary.get('strategy_files_count', 0)

        if test_count == 0:
            recommendations.append("Implement comprehensive test suite before production deployment")
        elif test_count < strategy_count:
            recommendations.append(f"Increase test coverage: Add tests for {strategy_count - test_count} strategy files")

        # Code quality recommendations
        quality_stats = summary.get('code_quality_stats', {})
        avg_lines = quality_stats.get('avg_lines_per_file', 0)
        if avg_lines > 400:
            recommendations.append("Refactor large files: Break down files over 400 lines into smaller modules")

        # Complexity recommendations
        complexity_dist = summary.get('complexity_distribution', {})
        high_complexity = complexity_dist.get('high', 0)
        if high_complexity > 0:
            recommendations.append(f"Reduce complexity: Refactor {high_complexity} high-complexity files")

        # Documentation recommendations
        doc_coverage = summary.get('documentation_coverage', {})
        poorly_doc = doc_coverage.get('poorly_documented', 0)
        if poorly_doc > 0:
            recommendations.append(f"Improve documentation: Add docstrings and comments to {poorly_doc} files")

        # Strategy implementation recommendations
        if strategy_count < 25:
            recommendations.append(f"Implement missing strategies: {25 - strategy_count} strategies need implementation")

        # Metrics and monitoring
        metric_files = summary.get('metric_files_count', 0)
        if metric_files < 5:
            recommendations.append("Enhance metrics collection: Implement comprehensive performance tracking")

        return recommendations

    async def generate_deep_dive_report(self, scan_results: Dict, output_dir: str = "reports/deep_dive") -> str:
        """Generate comprehensive deep dive report"""
        self.logger.info("Generating deep dive analysis report")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = output_path / f"deep_dive_report_{timestamp}.md"

        with open(report_path, 'w') as f:
            f.write("# Deep Dive File Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Executive Summary\n\n")
            f.write("Comprehensive analysis of system files for strategy-related data, ")
            f.write("metrics, and implementation quality.\n\n")

            # Summary statistics
            summary = scan_results.get('analysis_summary', {})
            f.write("## Analysis Summary\n\n")
            f.write(f"- **Total Files Analyzed:** {summary.get('total_files_analyzed', 0)}\n")
            f.write(f"- **Strategy Files:** {summary.get('strategy_files_count', 0)}\n")
            f.write(f"- **Test Files:** {summary.get('test_files_count', 0)}\n")
            f.write(f"- **Data Files:** {summary.get('data_files_count', 0)}\n")
            f.write(f"- **Config Files:** {summary.get('config_files_count', 0)}\n")
            f.write(f"- **Metric Files:** {summary.get('metric_files_count', 0)}\n\n")

            # Code quality stats
            quality_stats = summary.get('code_quality_stats', {})
            if quality_stats:
                f.write("### Code Quality Statistics\n\n")
                f.write(f"- Average Functions per File: {quality_stats.get('avg_functions_per_file', 0):.1f}\n")
                f.write(f"- Average Classes per File: {quality_stats.get('avg_classes_per_file', 0):.1f}\n")
                f.write(f"- Average Lines per File: {quality_stats.get('avg_lines_per_file', 0):.1f}\n\n")

            # Complexity distribution
            complexity_dist = summary.get('complexity_distribution', {})
            if complexity_dist:
                f.write("### Complexity Distribution\n\n")
                f.write(f"- Low Complexity: {complexity_dist.get('low', 0)} files\n")
                f.write(f"- Medium Complexity: {complexity_dist.get('medium', 0)} files\n")
                f.write(f"- High Complexity: {complexity_dist.get('high', 0)} files\n\n")

            # Documentation coverage
            doc_coverage = summary.get('documentation_coverage', {})
            if doc_coverage:
                f.write("### Documentation Coverage\n\n")
                f.write(f"- Well Documented: {doc_coverage.get('well_documented', 0)} files\n")
                f.write(f"- Moderately Documented: {doc_coverage.get('moderately_documented', 0)} files\n")
                f.write(f"- Poorly Documented: {doc_coverage.get('poorly_documented', 0)} files\n\n")

            # Key Insights
            insights = scan_results.get('insights', [])
            if insights:
                f.write("## Key Insights\n\n")
                for insight in insights:
                    f.write(f"- {insight}\n")
                f.write("\n")

            # Recommendations
            recommendations = scan_results.get('recommendations', [])
            if recommendations:
                f.write("## Recommendations\n\n")
                for rec in recommendations:
                    f.write(f"- {rec}\n")
                f.write("\n")

            # Detailed file analysis
            f.write("## Detailed File Analysis\n\n")

            all_files = (scan_results['strategy_files'] + scan_results['test_files'] +
                        scan_results['data_files'] + scan_results['config_files'])

            for file_info in all_files[:20]:  # Limit to top 20 files
                f.write(f"### {file_info['file_path']}\n\n")
                f.write(f"- **Size:** {file_info['file_size']} bytes\n")
                f.write(f"- **Last Modified:** {file_info['last_modified']}\n")
                f.write(f"- **Complexity Score:** {file_info.get('complexity_score', 0):.1f}\n")
                f.write(f"- **Documentation Score:** {file_info.get('documentation_score', 0):.1f}\n")

                # Metrics found
                metrics = file_info.get('metrics_found', {})
                if metrics:
                    f.write("- **Metrics Found:**\n")
                    for category, items in metrics.items():
                        f.write(f"  - {category}: {', '.join(items[:3])}\n")

                # Strategies found
                strategies = file_info.get('strategies_found', {})
                if strategies:
                    f.write("- **Strategies Found:**\n")
                    for category, items in strategies.items():
                        f.write(f"  - {category}: {', '.join(items[:3])}\n")

                f.write("\n")

        self.logger.info(f"Deep dive report generated: {report_path}")
        return str(report_path)

    async def find_strategy_dependencies(self, strategy_id: str) -> Dict[str, Any]:
        """Find all dependencies and related files for a strategy"""
        dependencies = {
            'strategy_id': strategy_id,
            'implementation_files': [],
            'test_files': [],
            'config_files': [],
            'data_files': [],
            'related_strategies': [],
            'external_dependencies': []
        }

        # Scan for files related to this strategy
        scan_results = await self.perform_comprehensive_scan()

        for file_info in scan_results['strategy_files']:
            if strategy_id.lower() in file_info['file_path'].lower():
                dependencies['implementation_files'].append(file_info)

        for file_info in scan_results['test_files']:
            if strategy_id.lower() in file_info['file_path'].lower():
                dependencies['test_files'].append(file_info)

        # Find related strategies (files that import or reference this strategy)
        for file_info in scan_results['strategy_files']:
            if strategy_id.lower() != file_info['file_path'].lower().split('/')[-1].split('.')[0]:
                content = ""
                try:
                    with open(PROJECT_ROOT / file_info['file_path'], 'r', encoding='utf-8') as f:
                        content = f.read()
                    if strategy_id.lower() in content.lower():
                        dependencies['related_strategies'].append(file_info)
                except:
                    pass

        return dependencies


# Global analyzer instance
deep_dive_analyzer = DeepDiveFileAnalyzer()


async def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Deep Dive File Analyzer")
    parser.add_argument('command', choices=['scan', 'report', 'dependencies'],
                       help='Command to execute')
    parser.add_argument('--strategy-id', help='Strategy ID for dependency analysis')
    parser.add_argument('--output-dir', default='reports/deep_dive', help='Output directory')

    args = parser.parse_args()

    if args.command == 'scan':
        print("[SCAN] Performing comprehensive system scan...")
        results = await deep_dive_analyzer.perform_comprehensive_scan()
        print(f"Scan completed: {results['total_files']} files analyzed")
        print(f"Strategy files: {len(results['strategy_files'])}")
        print(f"Test files: {len(results['test_files'])}")
        print(f"Metric files: {len(results['metric_files'])}")

    elif args.command == 'report':
        print("[REPORT] Generating deep dive report...")
        scan_results = await deep_dive_analyzer.perform_comprehensive_scan()
        report_path = await deep_dive_analyzer.generate_deep_dive_report(scan_results, args.output_dir)
        print(f"Report generated: {report_path}")

    elif args.command == 'dependencies':
        if not args.strategy_id:
            print("Strategy ID required for dependency analysis")
            return

        print(f"[DEPENDENCIES] Analyzing dependencies for {args.strategy_id}...")
        deps = await deep_dive_analyzer.find_strategy_dependencies(args.strategy_id)
        print(f"Implementation files: {len(deps['implementation_files'])}")
        print(f"Test files: {len(deps['test_files'])}")
        print(f"Related strategies: {len(deps['related_strategies'])}")


if __name__ == "__main__":
    asyncio.run(main())
