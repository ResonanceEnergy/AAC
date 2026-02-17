# NCC Enterprise Data Architecture - Data Schema Definitions

## Overview
This document defines the standardized data schemas for the NCC Enterprise Data Repository.
All data stored in the central repository must conform to these schemas to ensure consistency,
interoperability, and data quality across the enterprise.

## Core Data Categories

### 1. Agent Data Schema
Location: `_enterprise/data/agent_data/`
Purpose: Individual agent performance, status, and operational data

#### Agent Profile Schema (JSON)
```json
{
  "agent_id": "string (UUID)",
  "agent_name": "string",
  "division": "string",
  "department": "string",
  "role": "string",
  "status": "enum: active|inactive|suspended",
  "hire_date": "datetime",
  "last_activity": "datetime",
  "performance_score": "number (0-100)",
  "specializations": ["string"],
  "contact_info": {
    "email": "string",
    "slack_id": "string",
    "phone": "string"
  },
  "metadata": {
    "created_at": "datetime",
    "updated_at": "datetime",
    "version": "string"
  }
}
```

#### Agent Activity Schema (JSON)
```json
{
  "activity_id": "string (UUID)",
  "agent_id": "string (UUID)",
  "activity_type": "enum: task_completed|communication|system_interaction|error",
  "timestamp": "datetime",
  "duration_seconds": "number",
  "success": "boolean",
  "details": "object",
  "metadata": {
    "source_system": "string",
    "correlation_id": "string"
  }
}
```

### 2. Division Data Schema
Location: `_enterprise/data/division_data/`
Purpose: Division-level metrics, performance, and operational data

#### Division Metrics Schema (JSON)
```json
{
  "division_id": "string",
  "division_name": "string",
  "timestamp": "datetime",
  "metrics": {
    "agent_count": "number",
    "active_tasks": "number",
    "completed_tasks": "number",
    "success_rate": "number (0-100)",
    "average_response_time": "number (seconds)",
    "resource_utilization": "number (0-100)"
  },
  "kpis": {
    "efficiency_score": "number (0-100)",
    "quality_score": "number (0-100)",
    "innovation_index": "number (0-100)"
  },
  "metadata": {
    "reporting_period": "string",
    "data_quality_score": "number (0-100)"
  }
}
```

### 3. Enterprise Data Schema
Location: `_enterprise/data/enterprise_data/`
Purpose: Enterprise-wide metrics, analytics, and strategic data

#### Enterprise Dashboard Schema (JSON)
```json
{
  "timestamp": "datetime",
  "enterprise_metrics": {
    "total_agents": "number",
    "active_divisions": "number",
    "system_health_score": "number (0-100)",
    "overall_efficiency": "number (0-100)",
    "data_quality_index": "number (0-100)"
  },
  "division_breakdown": [
    {
      "division_name": "string",
      "contribution_percentage": "number",
      "health_status": "enum: excellent|good|warning|critical"
    }
  ],
  "alerts": [
    {
      "alert_id": "string",
      "severity": "enum: info|warning|error|critical",
      "message": "string",
      "timestamp": "datetime"
    }
  ],
  "metadata": {
    "data_freshness": "datetime",
    "last_updated": "datetime"
  }
}
```

### 4. Archive Data Schema
Location: `_enterprise/data/archives/`
Purpose: Historical data, backups, and long-term storage

#### Archive Metadata Schema (JSON)
```json
{
  "archive_id": "string (UUID)",
  "original_path": "string",
  "archive_date": "datetime",
  "retention_period_days": "number",
  "data_category": "enum: agent_data|division_data|enterprise_data",
  "compression_method": "string",
  "file_size_bytes": "number",
  "checksum": "string",
  "metadata": {
    "archived_by": "string",
    "reason": "string",
    "restoration_instructions": "string"
  }
}
```

## Data Quality Standards

### Required Fields
- All records must include `metadata.created_at` and `metadata.updated_at`
- All records must have a unique identifier
- Timestamps must be in ISO 8601 format
- Numeric values must be within defined ranges

### Data Validation Rules
- Agent IDs must be valid UUIDs
- Email addresses must be properly formatted
- Performance scores must be between 0-100
- Timestamps must not be in the future
- Required fields cannot be null or empty

### Data Types
- Strings: UTF-8 encoded, maximum 1000 characters unless specified
- Numbers: Standard IEEE 754 floating point
- Booleans: true/false only
- Arrays: Maximum 1000 elements
- Objects: Maximum nesting depth of 5 levels

## Version Control
- Schema version must be included in all records
- Breaking changes require version increment
- Backward compatibility must be maintained for 6 months

## Compliance
- All data must comply with NCC data governance policies
- PII must be encrypted and access-controlled
- Audit trails must be maintained for all data modifications