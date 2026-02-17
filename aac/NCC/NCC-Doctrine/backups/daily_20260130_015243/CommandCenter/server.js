import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Error handling - log but don't exit to maintain server stability
process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception (continuing):', err);
  // Don't exit - let the server continue running
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection (continuing):', promise, 'reason:', reason);
  // Don't exit - let the server continue running
});

const app = express();
const PORT = process.env.PORT || 4000;

app.use(cors());
app.use(express.json());

// Load NCC project schema
let projectSchema = {};
try {
  const schemaPath = path.join(__dirname, '..', 'NCC_ProjectSchema.json');
  const schemaContent = fs.readFileSync(schemaPath, 'utf8');
  projectSchema = JSON.parse(schemaContent);
  console.log('✅ Project schema loaded successfully');
} catch (error) {
  console.error('❌ Error loading project schema:', error.message);
  console.log('🔄 Using fallback project schema...');
  // Fallback schema with basic structure
  projectSchema = {
    projects: [
      {
        name: "Resonance Energy",
        slug: "resonance-energy",
        category: "Energy",
        path: "ResonanceEnergyCorp",
        status: "Active",
        progress: 35,
        azParticipates: true,
        cSuiteParticipates: true
      },
      {
        name: "Big Brain Intelligence",
        slug: "big-brain-intelligence",
        category: "R&D",
        path: "BigBrainIntelligence",
        status: "Planning",
        progress: 10,
        azParticipates: true,
        cSuiteParticipates: true
      }
    ]
  };
}

// Initialize task management system
let departmentTasks = {};
let securityIncidents = [];
let cSuiteOversight = {
  lastUpdate: new Date().toISOString(),
  activeAlerts: 0,
  complianceScore: 100,
  departmentsMonitored: 0
};

// Initialize tasks for all departments
function initializeDepartmentTasks() {
  if (projectSchema.projects) {
    projectSchema.projects.forEach(project => {
      const deptId = project.slug;
      departmentTasks[deptId] = {
        department: project.name,
        ceo: `${project.name} CEO`,
        lastUpdate: new Date().toISOString(),
        security10Compliance: {
          status: 'COMPLIANT',
          lastAudit: new Date().toISOString(),
          incidents: 0,
          containmentIntegrity: 100
        },
        tasks: [
          {
            id: `security-10-${deptId}`,
            title: 'Implement Security 10 Directive',
            description: 'Deploy zero-tolerance containment protocols for all cross-contamination, leaks, and spills',
            priority: 'CRITICAL',
            status: 'ACTIVE',
            assignee: `${project.name} CEO`,
            dueDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
            created: new Date().toISOString(),
            category: 'SECURITY',
            dependencies: [],
            progress: 0,
            subtasks: [
              { id: 'security-infrastructure', title: 'Deploy containment monitoring systems', completed: false },
              { id: 'personnel-training', title: 'Complete Security 10 certification for all staff', completed: false },
              { id: 'access-controls', title: 'Implement multi-factor authentication', completed: false },
              { id: 'audit-systems', title: 'Establish automated audit trails', completed: false }
            ]
          },
          {
            id: `nnc-compliance-${deptId}`,
            title: 'NNO Compartmentalization Implementation',
            description: 'Ensure all operations comply with Need-to-Know-Only security protocols',
            priority: 'HIGH',
            status: 'ACTIVE',
            assignee: `${project.name} CEO`,
            dueDate: new Date(Date.now() + 45 * 24 * 60 * 60 * 1000).toISOString(),
            created: new Date().toISOString(),
            category: 'COMPLIANCE',
            dependencies: [`security-10-${deptId}`],
            progress: 0,
            subtasks: [
              { id: 'compartment-mapping', title: 'Map all information compartments', completed: false },
              { id: 'access-controls', title: 'Implement compartment access controls', completed: false },
              { id: 'training-program', title: 'Deploy NNO training program', completed: false },
              { id: 'audit-compliance', title: 'Pass initial NNO audit', completed: false }
            ]
          },
          {
            id: `operational-readiness-${deptId}`,
            title: 'Department Operational Readiness',
            description: 'Ensure department is fully operational and integrated with NCC systems',
            priority: 'HIGH',
            status: 'PENDING',
            assignee: `${project.name} CEO`,
            dueDate: new Date(Date.now() + 60 * 24 * 60 * 60 * 1000).toISOString(),
            created: new Date().toISOString(),
            category: 'OPERATIONS',
            dependencies: [`security-10-${deptId}`, `nnc-compliance-${deptId}`],
            progress: 0,
            subtasks: [
              { id: 'infrastructure-setup', title: 'Complete department infrastructure', completed: false },
              { id: 'team-formation', title: 'Assemble core team', completed: false },
              { id: 'system-integration', title: 'Integrate with NCC dashboard', completed: false },
              { id: 'performance-metrics', title: 'Establish KPIs and reporting', completed: false }
            ]
          }
        ]
      };
    });
  }
  cSuiteOversight.departmentsMonitored = Object.keys(departmentTasks).length;
}

// Initialize on startup
initializeDepartmentTasks();

// Auto-evolution system
setInterval(() => {
  const now = new Date();

  // Update task statuses based on dependencies and time
  Object.keys(departmentTasks).forEach(deptId => {
    const dept = departmentTasks[deptId];
    dept.tasks.forEach(task => {
      // Check dependencies
      if (task.dependencies.length > 0) {
        const depsMet = task.dependencies.every(depId => {
          const depTask = dept.tasks.find(t => t.id === depId);
          return depTask && depTask.status === 'COMPLETED';
        });
        if (depsMet && task.status === 'PENDING') {
          task.status = 'ACTIVE';
        }
      }

      // Auto-progress based on time (simulated evolution)
      if (task.status === 'ACTIVE' && Math.random() < 0.1) { // 10% chance per interval
        task.progress = Math.min(100, task.progress + Math.floor(Math.random() * 10));
        if (task.progress >= 100) {
          task.status = 'COMPLETED';
          task.completedDate = now.toISOString();
        }
      }

      // Update subtasks
      task.subtasks.forEach(subtask => {
        if (!subtask.completed && Math.random() < 0.05) { // 5% chance
          subtask.completed = true;
        }
      });
    });

    dept.lastUpdate = now.toISOString();
  });

  // Update C-Suite oversight
  cSuiteOversight.lastUpdate = now.toISOString();
  cSuiteOversight.activeAlerts = securityIncidents.filter(i => i.status === 'ACTIVE').length;
  cSuiteOversight.complianceScore = Math.max(0, 100 - securityIncidents.length * 2);
}, 30000); // Update every 30 seconds

// Placeholder data
let priorities = [
  'Maintain comms redundancy',
  'EMP readiness',
  'Critical project: Avatar Center'
];
let analytics = {
  activeProjects: 54,
  alerts: 0,
  opsPackets: 7
};
let timeline = [
  { event: 'System boot', time: '2026-01-27T08:00:00Z' },
  { event: 'EMP readiness check', time: '2026-01-27T09:00:00Z' }
];

// API endpoints
app.get('/api/priorities', (req, res) => res.json({ priorities }));
app.get('/api/analytics', (req, res) => res.json(analytics));
app.get('/api/timeline', (req, res) => res.json({ timeline }));

// Task Management API
app.get('/api/tasks', (req, res) => {
  res.json({
    departments: departmentTasks,
    summary: {
      totalDepartments: Object.keys(departmentTasks).length,
      activeTasks: Object.values(departmentTasks).reduce((sum, dept) =>
        sum + dept.tasks.filter(t => t.status === 'ACTIVE').length, 0),
      completedTasks: Object.values(departmentTasks).reduce((sum, dept) =>
        sum + dept.tasks.filter(t => t.status === 'COMPLETED').length, 0),
      criticalTasks: Object.values(departmentTasks).reduce((sum, dept) =>
        sum + dept.tasks.filter(t => t.priority === 'CRITICAL').length, 0)
    }
  });
});

app.get('/api/tasks/:department', (req, res) => {
  const dept = departmentTasks[req.params.department];
  if (!dept) {
    return res.status(404).json({ error: 'Department not found' });
  }
  res.json(dept);
});

app.post('/api/tasks/:department/:taskId/update', (req, res) => {
  const { department, taskId } = req.params;
  const { status, progress, notes } = req.body;

  const dept = departmentTasks[department];
  if (!dept) {
    return res.status(404).json({ error: 'Department not found' });
  }

  const task = dept.tasks.find(t => t.id === taskId);
  if (!task) {
    return res.status(404).json({ error: 'Task not found' });
  }

  if (status) task.status = status;
  if (progress !== undefined) task.progress = Math.max(0, Math.min(100, progress));
  if (notes) task.notes = notes;

  task.lastModified = new Date().toISOString();

  // Auto-complete if progress reaches 100
  if (task.progress >= 100 && task.status !== 'COMPLETED') {
    task.status = 'COMPLETED';
    task.completedDate = new Date().toISOString();
  }

  res.json({ success: true, task });
});

// Security 10 Incident Reporting
app.post('/api/security-incident', (req, res) => {
  const incident = {
    id: `incident-${Date.now()}`,
    ...req.body,
    reportedAt: new Date().toISOString(),
    status: 'ACTIVE',
    severity: req.body.category === 'A' ? 'CRITICAL' : req.body.category === 'B' ? 'HIGH' : 'MEDIUM'
  };

  securityIncidents.push(incident);

  // Update department compliance
  if (incident.department && departmentTasks[incident.department]) {
    departmentTasks[incident.department].security10Compliance.incidents++;
    departmentTasks[incident.department].security10Compliance.containmentIntegrity =
      Math.max(0, departmentTasks[incident.department].security10Compliance.containmentIntegrity - 10);
  }

  res.json({ success: true, incidentId: incident.id });
});

app.get('/api/security-incidents', (req, res) => {
  res.json({
    incidents: securityIncidents,
    summary: {
      total: securityIncidents.length,
      active: securityIncidents.filter(i => i.status === 'ACTIVE').length,
      critical: securityIncidents.filter(i => i.severity === 'CRITICAL').length,
      resolved: securityIncidents.filter(i => i.status === 'RESOLVED').length
    }
  });
});

// C-Suite Oversight Dashboard
app.get('/api/c-suite-oversight', (req, res) => {
  const oversight = {
    ...cSuiteOversight,
    departmentStatus: Object.keys(departmentTasks).map(deptId => ({
      department: departmentTasks[deptId].department,
      ceo: departmentTasks[deptId].ceo,
      compliance: departmentTasks[deptId].security10Compliance,
      activeTasks: departmentTasks[deptId].tasks.filter(t => t.status === 'ACTIVE').length,
      criticalTasks: departmentTasks[deptId].tasks.filter(t => t.priority === 'CRITICAL' && t.status !== 'COMPLETED').length,
      lastUpdate: departmentTasks[deptId].lastUpdate
    }))
  };
  res.json(oversight);
});

// Auto-generate new tasks based on system evolution
app.post('/api/tasks/evolve', (req, res) => {
  const { department, taskType, description } = req.body;

  if (!departmentTasks[department]) {
    return res.status(404).json({ error: 'Department not found' });
  }

  const newTask = {
    id: `${taskType}-${department}-${Date.now()}`,
    title: `${taskType.replace('-', ' ').toUpperCase()} Task`,
    description: description || `Automated ${taskType} task for ${departmentTasks[department].department}`,
    priority: 'MEDIUM',
    status: 'ACTIVE',
    assignee: departmentTasks[department].ceo,
    dueDate: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
    created: new Date().toISOString(),
    category: taskType.toUpperCase(),
    dependencies: [],
    progress: 0,
    subtasks: []
  };

  departmentTasks[department].tasks.push(newTask);
  res.json({ success: true, task: newTask });
});

app.listen(PORT, (err) => {
  if (err) {
    console.error('Failed to start server:', err);
    process.exit(1);
  }
  console.log(`🚀 NCC Command Center API running at http://localhost:${PORT}`);
  console.log(`📊 Task Management System: ACTIVE`);
  console.log(`🔒 Security 10 Directive: ENFORCED`);
  console.log(`🏢 Departments Monitored: ${Object.keys(departmentTasks).length}`);
  console.log(`🔄 Auto-evolution cycle: RUNNING`);
  console.log(`🌐 Ready to accept connections...`);
});
