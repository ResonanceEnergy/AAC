<# 
.SYNOPSIS
    Theater C - Infrastructure Arbitrage Operations
    
.DESCRIPTION
    Implements the Infrastructure/Latency theater for detecting and exploiting
    technical inefficiencies across exchanges and networks.
    
.NOTES
    Part of ACC Doctrine Implementation
    Version: 1.0.0
#>

param(
    [switch]$Verbose,
    [switch]$DryRun,
    [string]$ConfigPath = "$PSScriptRoot\Doctrine_Config.yaml"
)

$ErrorActionPreference = "Stop"

# ============================================
# CONFIGURATION
# ============================================

$script:TheaterConfig = @{
    Name = "Theater_C"
    Description = "Infrastructure Arbitrage"
    Agents = @(
        "latency_monitor",
        "bridge_analyzer",
        "gas_optimizer",
        "liquidity_tracker"
    )
    RefreshIntervalSeconds = 60  # Faster refresh for infra signals
    ConfidenceThreshold = 0.70
    MaxConcurrentSignals = 5
    LatencyThresholdMs = 100
    MinArbitrageSpreadPct = 0.5
}

# ============================================
# LOGGING
# ============================================

$LogDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

function Write-TheaterLog {
    param([string]$Message, [string]$Level = "INFO")
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] [Theater_C] $Message"
    
    $logFile = Join-Path $LogDir "theater_c_$(Get-Date -Format 'yyyyMMdd').log"
    Add-Content -Path $logFile -Value $logEntry
    
    if ($Verbose -or $Level -eq "ERROR") {
        switch ($Level) {
            "ERROR" { Write-Host $logEntry -ForegroundColor Red }
            "WARN"  { Write-Host $logEntry -ForegroundColor Yellow }
            "INFO"  { Write-Host $logEntry -ForegroundColor Green }
            default { Write-Host $logEntry }
        }
    }
}

# ============================================
# TRIGGER DETECTION
# ============================================

class InfraSignal {
    [string]$SignalId
    [string]$SignalType
    [string]$Target       # Exchange, network, or bridge
    [double]$Value        # Latency, spread, or gas
    [double]$Confidence
    [string]$Source
    [datetime]$DetectedAt
    [int]$TTLSeconds      # Time to live - infra signals expire fast
    [hashtable]$Metadata
    
    InfraSignal([string]$type, [string]$target, [double]$value, [double]$confidence, [int]$ttl = 300) {
        $this.SignalId = [guid]::NewGuid().ToString().Substring(0,8)
        $this.SignalType = $type
        $this.Target = $target
        $this.Value = $value
        $this.Confidence = $confidence
        $this.TTLSeconds = $ttl
        $this.DetectedAt = Get-Date
        $this.Metadata = @{}
    }
    
    [bool] IsExpired() {
        return (Get-Date) -gt $this.DetectedAt.AddSeconds($this.TTLSeconds)
    }
}

function Invoke-TriggerDetection {
    <#
    .SYNOPSIS
        Detects infrastructure-based trading triggers
    #>
    
    Write-TheaterLog "Starting infrastructure scan..."
    
    $triggers = @()
    
    # 1. Check latency conditions
    $latencySignals = Get-LatencySignals
    foreach ($signal in $latencySignals) {
        if (-not $signal.IsExpired() -and $signal.Confidence -ge $script:TheaterConfig.ConfidenceThreshold) {
            $triggers += $signal
            Write-TheaterLog "Latency trigger: $($signal.Target) - $($signal.Value)ms" "INFO"
        }
    }
    
    # 2. Check bridge arbitrage opportunities
    $bridgeSignals = Get-BridgeSignals
    foreach ($signal in $bridgeSignals) {
        if (-not $signal.IsExpired() -and $signal.Value -ge $script:TheaterConfig.MinArbitrageSpreadPct) {
            $triggers += $signal
            Write-TheaterLog "Bridge arb trigger: $($signal.Target) spread=$($signal.Value)%" "INFO"
        }
    }
    
    # 3. Check gas optimization windows
    $gasSignals = Get-GasSignals
    foreach ($signal in $gasSignals) {
        if (-not $signal.IsExpired() -and $signal.Confidence -ge $script:TheaterConfig.ConfidenceThreshold) {
            $triggers += $signal
            Write-TheaterLog "Gas trigger: $($signal.Target) - $($signal.Value) gwei" "INFO"
        }
    }
    
    # 4. Check liquidity imbalances
    $liquiditySignals = Get-LiquiditySignals
    foreach ($signal in $liquiditySignals) {
        if (-not $signal.IsExpired() -and $signal.Confidence -ge $script:TheaterConfig.ConfidenceThreshold) {
            $triggers += $signal
            Write-TheaterLog "Liquidity trigger: $($signal.Target) imbalance=$($signal.Value)" "INFO"
        }
    }
    
    Write-TheaterLog "Infrastructure scan complete: $($triggers.Count) triggers detected"
    
    return $triggers
}

function Get-LatencySignals {
    <#
    .SYNOPSIS
        Calls Python latency monitor agent
    #>
    
    $signals = @()
    
    try {
        $pythonScript = Join-Path $PSScriptRoot ".." "BigBrainIntelligence" "agents.py"
        
        if (Test-Path $pythonScript) {
            $result = python -c @"
import sys
sys.path.insert(0, r'$((Join-Path $PSScriptRoot "..").Replace('\', '\\'))')
import asyncio
from BigBrainIntelligence.agents import get_agent
agent = get_agent('latency_monitor')
if agent:
    findings = asyncio.run(agent.run_scan())
    for f in findings:
        print(f'{f.data.get("exchange", "unknown")}|{f.data.get("latency_ms", 0)}|{f.confidence}')
"@
            
            foreach ($line in $result -split "`n") {
                if ($line.Trim()) {
                    $parts = $line.Split('|')
                    if ($parts.Count -ge 3) {
                        $signal = [InfraSignal]::new(
                            "latency_anomaly",
                            $parts[0],
                            [double]$parts[1],
                            [double]$parts[2],
                            120  # 2 minute TTL for latency signals
                        )
                        $signal.Source = "latency_monitor"
                        $signals += $signal
                    }
                }
            }
        }
    }
    catch {
        Write-TheaterLog "Latency analysis error: $_" "ERROR"
    }
    
    return $signals
}

function Get-BridgeSignals {
    <#
    .SYNOPSIS
        Calls Python bridge analyzer agent
    #>
    
    $signals = @()
    
    try {
        $pythonScript = Join-Path $PSScriptRoot ".." "BigBrainIntelligence" "agents.py"
        
        if (Test-Path $pythonScript) {
            $result = python -c @"
import sys
sys.path.insert(0, r'$((Join-Path $PSScriptRoot "..").Replace('\', '\\'))')
import asyncio
from BigBrainIntelligence.agents import get_agent
agent = get_agent('bridge_analyzer')
if agent:
    findings = asyncio.run(agent.run_scan())
    for f in findings:
        print(f'{f.data.get("bridge", "unknown")}|{f.data.get("spread_pct", 0)}|{f.confidence}')
"@
            
            foreach ($line in $result -split "`n") {
                if ($line.Trim()) {
                    $parts = $line.Split('|')
                    if ($parts.Count -ge 3) {
                        $signal = [InfraSignal]::new(
                            "bridge_arbitrage",
                            $parts[0],
                            [double]$parts[1],
                            [double]$parts[2],
                            180  # 3 minute TTL for bridge arb
                        )
                        $signal.Source = "bridge_analyzer"
                        $signals += $signal
                    }
                }
            }
        }
    }
    catch {
        Write-TheaterLog "Bridge analysis error: $_" "ERROR"
    }
    
    return $signals
}

function Get-GasSignals {
    <#
    .SYNOPSIS
        Calls Python gas optimizer agent
    #>
    
    $signals = @()
    
    try {
        $pythonScript = Join-Path $PSScriptRoot ".." "BigBrainIntelligence" "agents.py"
        
        if (Test-Path $pythonScript) {
            $result = python -c @"
import sys
sys.path.insert(0, r'$((Join-Path $PSScriptRoot "..").Replace('\', '\\'))')
import asyncio
from BigBrainIntelligence.agents import get_agent
agent = get_agent('gas_optimizer')
if agent:
    findings = asyncio.run(agent.run_scan())
    for f in findings:
        print(f'{f.data.get("network", "unknown")}|{f.data.get("current_gwei", 0)}|{f.confidence}')
"@
            
            foreach ($line in $result -split "`n") {
                if ($line.Trim()) {
                    $parts = $line.Split('|')
                    if ($parts.Count -ge 3) {
                        $signal = [InfraSignal]::new(
                            "gas_opportunity",
                            $parts[0],
                            [double]$parts[1],
                            [double]$parts[2],
                            300  # 5 minute TTL for gas
                        )
                        $signal.Source = "gas_optimizer"
                        $signals += $signal
                    }
                }
            }
        }
    }
    catch {
        Write-TheaterLog "Gas analysis error: $_" "ERROR"
    }
    
    return $signals
}

function Get-LiquiditySignals {
    <#
    .SYNOPSIS
        Calls Python liquidity tracker agent
    #>
    
    $signals = @()
    
    try {
        $pythonScript = Join-Path $PSScriptRoot ".." "BigBrainIntelligence" "agents.py"
        
        if (Test-Path $pythonScript) {
            $result = python -c @"
import sys
sys.path.insert(0, r'$((Join-Path $PSScriptRoot "..").Replace('\', '\\'))')
import asyncio
from BigBrainIntelligence.agents import get_agent
agent = get_agent('liquidity_tracker')
if agent:
    findings = asyncio.run(agent.run_scan())
    for f in findings:
        print(f'{f.data.get("symbol", "unknown")}|{f.data.get("imbalance_ratio", 0)}|{f.confidence}')
"@
            
            foreach ($line in $result -split "`n") {
                if ($line.Trim()) {
                    $parts = $line.Split('|')
                    if ($parts.Count -ge 3) {
                        $signal = [InfraSignal]::new(
                            "liquidity_imbalance",
                            $parts[0],
                            [double]$parts[1],
                            [double]$parts[2],
                            240  # 4 minute TTL for liquidity
                        )
                        $signal.Source = "liquidity_tracker"
                        $signals += $signal
                    }
                }
            }
        }
    }
    catch {
        Write-TheaterLog "Liquidity analysis error: $_" "ERROR"
    }
    
    return $signals
}

# ============================================
# ACTION EXECUTION
# ============================================

function Invoke-TheaterAction {
    param(
        [Parameter(Mandatory)]
        [InfraSignal]$Signal
    )
    
    # Check if signal has expired
    if ($Signal.IsExpired()) {
        Write-TheaterLog "Signal $($Signal.SignalId) expired, skipping" "WARN"
        return
    }
    
    Write-TheaterLog "Executing action for signal: $($Signal.SignalId) - $($Signal.SignalType)"
    
    if ($DryRun) {
        Write-TheaterLog "[DRY RUN] Would execute: $($Signal.SignalType) on $($Signal.Target)" "INFO"
        return
    }
    
    # Determine action based on signal type
    switch ($Signal.SignalType) {
        "latency_anomaly" {
            # Exploit latency advantage
            $action = @{
                Type = "LATENCY_ARB"
                Exchange = $Signal.Target
                ExpectedAdvantageMs = $Signal.Value
                Priority = "HIGH"
                Reason = "Latency anomaly detected"
            }
        }
        "bridge_arbitrage" {
            # Execute cross-chain arbitrage
            $action = @{
                Type = "BRIDGE_ARB"
                Bridge = $Signal.Target
                SpreadPct = $Signal.Value
                Priority = "CRITICAL"
                Reason = "Bridge arbitrage opportunity"
            }
        }
        "gas_opportunity" {
            # Queue transactions for low gas
            $action = @{
                Type = "GAS_QUEUE"
                Network = $Signal.Target
                CurrentGwei = $Signal.Value
                Priority = "MEDIUM"
                Reason = "Low gas window"
            }
        }
        "liquidity_imbalance" {
            # Exploit liquidity difference
            $action = @{
                Type = "LIQUIDITY_ARB"
                Symbol = $Signal.Target
                ImbalanceRatio = $Signal.Value
                Priority = "HIGH"
                Reason = "Liquidity imbalance"
            }
        }
        default {
            Write-TheaterLog "Unknown signal type: $($Signal.SignalType)" "WARN"
            return
        }
    }
    
    # Log the action
    $actionLog = Join-Path $LogDir "actions_$(Get-Date -Format 'yyyyMMdd').json"
    $actionEntry = @{
        Timestamp = (Get-Date).ToString("o")
        Signal = @{
            Id = $Signal.SignalId
            Type = $Signal.SignalType
            Target = $Signal.Target
            Value = $Signal.Value
            Confidence = $Signal.Confidence
            TTL = $Signal.TTLSeconds
        }
        Action = $action
    }
    
    $existingActions = @()
    if (Test-Path $actionLog) {
        $existingActions = Get-Content $actionLog | ConvertFrom-Json
    }
    $existingActions += $actionEntry
    $existingActions | ConvertTo-Json -Depth 10 | Set-Content $actionLog
    
    Write-TheaterLog "Action logged: $($action.Type) on $($Signal.Target) (priority: $($action.Priority))"
}

# ============================================
# MAIN LOOP
# ============================================

function Start-TheaterC {
    param(
        [switch]$Continuous
    )
    
    Write-TheaterLog "=== Theater C Starting ===" "INFO"
    Write-TheaterLog "Agents: $($script:TheaterConfig.Agents -join ', ')"
    Write-TheaterLog "Confidence Threshold: $($script:TheaterConfig.ConfidenceThreshold)"
    Write-TheaterLog "Min Arbitrage Spread: $($script:TheaterConfig.MinArbitrageSpreadPct)%"
    
    do {
        try {
            # Detect triggers
            $triggers = Invoke-TriggerDetection
            
            # Sort by priority (higher value = higher priority for infra signals)
            $sortedTriggers = $triggers | Sort-Object -Property Value -Descending
            
            # Process triggers (limited to max concurrent)
            $processCount = [Math]::Min($sortedTriggers.Count, $script:TheaterConfig.MaxConcurrentSignals)
            
            for ($i = 0; $i -lt $processCount; $i++) {
                Invoke-TheaterAction -Signal $sortedTriggers[$i]
            }
            
            if ($Continuous) {
                Write-TheaterLog "Sleeping for $($script:TheaterConfig.RefreshIntervalSeconds)s..."
                Start-Sleep -Seconds $script:TheaterConfig.RefreshIntervalSeconds
            }
        }
        catch {
            Write-TheaterLog "Theater loop error: $_" "ERROR"
            if ($Continuous) {
                Start-Sleep -Seconds 30
            }
        }
    } while ($Continuous)
    
    Write-TheaterLog "=== Theater C Stopped ===" "INFO"
}

# Entry point
if ($MyInvocation.InvocationName -ne '.') {
    Start-TheaterC
}
