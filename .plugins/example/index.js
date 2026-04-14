// This is an example plugin that registers a new Node, Activation Function, Learning Rule, Error Function, Synapse, and Model.
// It leverages window.React since it is dynamically evaluated as a blob in the browser.

const React = window.React;
const { Handle, Position } = window.ReactFlow;

// --- 1. Example Activation Function (Standard/Non-Differentiable) ---
class ExampleStepActivationFunction {
    constructor() {
        this.name = 'example-step';
    }
    
    evaluate(x) { return x >= 0 ? 1 : -1; }

    formulas(xExpr = 'x') { return [`f(${xExpr}) = \\begin{cases} 1 & ${xExpr} \\ge 0 \\\\ -1 & ${xExpr} < 0 \\end{cases}`]; }
    derivativeFormulas(xExpr = 'x') { return [`f'(${xExpr}) = 0`]; }
}

// --- 1.5. Example Activation Function (Differentiable) ---
class ExampleLeakyActivationFunction {
    constructor() {
        this.name = 'example-leaky';
    }
    
    evaluate(x) { return x > 0 ? x * 1.5 : x * 0.1; }
    derivative(x) { return x > 0 ? 1.5 : 0.1; }

    formulas(xExpr = 'x') { return [`f(${xExpr}) = \\begin{cases} 1.5 \\cdot ${xExpr} & ${xExpr} > 0 \\\\ 0.1 \\cdot ${xExpr} & ${xExpr} \\le 0 \\end{cases}`]; }
    derivativeFormulas(xExpr = 'x') { return [`f'(${xExpr}) = \\begin{cases} 1.5 & ${xExpr} > 0 \\\\ 0.1 & ${xExpr} \\le 0 \\end{cases}`]; }
}

// --- 2. Example Learning Rule ---
class ExampleLearningRule {
    constructor() {
        this.name = 'example-learning-rule';
    }

    formatFormula() {
        return '\\Delta w_i = n \\cdot e \\cdot |e| \\cdot x_i';
    }

    getDefaultActivation() {
        return 'example-leaky';
    }

    calculateClassificationOutput(output, netInput, desired) {
        // Optional hook: custom thresholding for Metrics & Timeline UI
        if (desired === 1 || desired === -1) return netInput >= 0 ? 1 : -1;
        if (desired === 1 || desired === 0) return netInput >= 0 ? 1 : 0;
        return netInput;
    }

    requiresContinuousConvergence(errorAlgorithm) {
        // Optional hook: True if this rule trains towards minimum continuous Loss instead of step accuracy.
        return errorAlgorithm === 'lms';
    }

    applyRule(neuron, synapses, inputValues, desired, actual, netInput, lr) {
        // Standard supervised learning rule implementation from the new Architecture
        const incomingSynapses = synapses.filter(s => s.postSynaptic.id === neuron.id);
        const error = desired - actual;
        
        for (const syn of incomingSynapses) {
            let xVal = inputValues[syn.preSynaptic.id] ?? syn.preSynaptic.output;
            if (Array.isArray(xVal)) xVal = xVal[syn.sourceIndex !== undefined ? syn.sourceIndex : 0] ?? 0;
            
            // Example rule: n * error * |error| * x
            const delta = lr * error * Math.abs(error) * (typeof xVal === 'number' ? xVal : 0);
            
            syn.weight += delta;
            
            // Mirror bias back to the neuron object if updating a bias synapse
            if (syn.targetHandle === 'bias' && 'bias' in neuron) {
                neuron.bias = syn.weight;
            }
        }
    }
}

// --- 3. Example Error Function ---
class ExampleErrorFunction {
    constructor(params = {}) {
        this.name = 'example-error-function';
        this.delta = params.delta !== undefined ? params.delta : 1.0;
    }

    calculateError(target, output) {
        const a = Math.abs(target - output);
        if (a <= this.delta) {
            return 0.5 * a * a;
        } else {
            return this.delta * (a - 0.5 * this.delta);
        }
    }

    calculateDerivative(target, output) {
        const diff = target - output;
        if (Math.abs(diff) <= this.delta) {
            return -diff;
        } else {
            return -this.delta * Math.sign(diff);
        }
    }

    aggregateError(totalError, numSamples) {
        return numSamples > 0 ? totalError / numSamples : 0;
    }

    formatSurfaceFormula(xLabel, yLabel, yHatExpr) {
        return `J(${xLabel}, ${yLabel}) = Σ Huber(d, ${yHatExpr}, δ=${this.delta})`;
    }
}

// --- 4. Example Model State ---
class ExampleNodeState {
    constructor(id) {
        this.id = id;
        this.type = 'example-node';
        this.label = 'Example Model';
        this.output = 0;
        this.netInput = 0;
        this.bias = 0;
        this.activationFunction = 'example-leaky'; // Uses differentiable version
        this.learningRule = 'example-learning-rule';
    }
}

// --- 4. Example Model (Flyweight) ---
class ExampleModel {
    constructor() {
        this.type = 'example-node';
    }

    getSize(state) {
        return 1;
    }

    getPropertiesView(state) {
        return { 
            name: 'example-ui-panel', 
            params: { showExtraInfo: true, customMessage: "This text is passed through the dynamic PropertyRegistry params." } 
        };
    }

    calculateOutput(state, incomingSynapses = []) {
        let sum = 0;
        incomingSynapses.forEach(synapse => {
            if (synapse && synapse.preSynaptic) {
                const inVal = synapse.preSynaptic.output;
                const weight = synapse.weight || 1;
                sum += (Array.isArray(inVal) ? (inVal[0] ?? 0) : inVal) * weight;
            }
        });
        state.output = sum > 0 ? sum * 1.5 : sum * 0.1;
        state.netInput = sum;
        return state.output;
    }

    trainForward(state, incomingSynapses = [], context) {
        let net = 0;
        incomingSynapses.forEach(synapse => {
            const inVal = context.resolveInput(synapse);
            const weight = synapse.weight || 1;
            net += (Array.isArray(inVal) ? (inVal[0] ?? 0) : inVal) * weight;
        });
        const outVal = net > 0 ? net * 1.5 : net * 0.1;
        state.output = outVal;
        state.netInput = net;
        return { output: outVal, netInput: net };
    }

    static execute(node, incomingEdges, ctx) {
        let sum = 0;
        incomingEdges.forEach(edge => {
            const outArr = ctx.getSourceOutput(edge);
            const inVal = outArr.length > 0 ? outArr[0] : 0;
            const weight = edge.weight || 1;
            sum += inVal * weight;
        });
        const outVal = sum > 0 ? sum * 1.5 : sum * 0.1;
        node.output = [outVal];
        node.netInput = sum;
        return node;
    }
}

// --- 5. Example Synapse ---
class ExampleSynapse {
    constructor(id, pre, post) {
        this.id = id;
        this.preSynaptic = pre;
        this.postSynaptic = post;
        this.weight = Math.random() * 2 - 1;
        this.type = 'example-synapse';
    }
}

// --- 6. Example Node React Component ---
const ExampleNodeComponent = ({ data, selected }) => {
    return React.createElement(
        'div',
        {
            className: `px-4 py-2 rounded-lg border-2 shadow-md bg-white dark:bg-neutral-800 ${selected ? 'border-amber-500' : 'border-amber-300 dark:border-amber-600'} min-w-[120px] text-center`
        },
        [
            React.createElement(Handle, {
                key: 'handle-target',
                id: 'input',
                type: 'target',
                position: Position.Left,
                className: 'w-3 h-3 bg-amber-500 border border-white dark:border-neutral-800'
            }),
            React.createElement('div', { key: 'title', className: 'text-sm font-bold text-amber-600 dark:text-amber-400' }, 'Example Plugin Node'),
            React.createElement('div', { key: 'label', className: 'text-xs text-gray-500' }, data.state?.label || data.neuron?.label || data.label || 'Example'),
            React.createElement(Handle, {
                key: 'handle-source',
                id: 'output',
                type: 'source',
                position: Position.Right,
                className: 'w-3 h-3 bg-amber-500 border border-white dark:border-neutral-800'
            })
        ]
    );
};

// --- 7. Example Properties Panel Component ---
const ExamplePropertiesComponent = ({ selectedNode, onUpdateNeuron, showExtraInfo, customMessage }) => {
    return React.createElement(
        'div',
        { className: 'flex flex-col gap-2 mt-4' },
        [
            React.createElement('div', { key: 'header', className: 'text-xs uppercase font-bold text-amber-500' }, 'Example Settings'),
            React.createElement('div', { key: 'desc', className: 'text-xs text-slate-500' }, 'This node was injected by the example plugin at runtime.'),
            showExtraInfo ? React.createElement('div', { key: 'extra', className: 'mt-2 p-2 bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded text-xs italic' }, customMessage) : null
        ]
    );
};

// --- 8. Example Initializer ---
class ExampleInitializer {
    constructor(params = {}) {
        this.name = 'example-initializer';
        this.fillValue = params.fillValue !== undefined ? params.fillValue : 0.123;
    }

    apply(shape) {
        if (window.tf) {
            return window.tf.fill(shape, this.fillValue);
        }
        return { shape, isFallback: true }; 
    }

    formulas() {
        return [`W = \\text{fill}(${this.fillValue})`];
    }
}

// --- 9. Example Normalization ---
class ExampleWinsorizeNormalization {
    constructor(params = {}) {
        this.name = 'example-winsorize';
        this.label = 'Example Winsorize (P5/P95)';
    }

    normalize(data, numCols) {
        if (!data || data.length === 0 || numCols <= 0) return data.slice();
        const numRows = Math.floor(data.length / numCols);
        const result = data.slice();

        for (let c = 0; c < numCols; c++) {
            const col = [];
            for (let r = 0; r < numRows; r++) {
                col.push(data[r * numCols + c]);
            }
            col.sort((a, b) => a - b);
            
            const p5Idx = Math.floor(0.05 * (col.length - 1));
            const p95Idx = Math.ceil(0.95 * (col.length - 1));
            const minVal = col[p5Idx];
            const maxVal = col[p95Idx];

            for (let r = 0; r < numRows; r++) {
                let val = data[r * numCols + c];
                if (val < minVal) val = minVal;
                if (val > maxVal) val = maxVal;
                result[r * numCols + c] = val;
            }
        }
        return result;
    }

    formatFormula(xExpr = 'x') {
        return `f(${xExpr}) = \\max(P_{5}, \\min(${xExpr}, P_{95}))`;
    }
}

// --- 10. Example Metric ---
class ExampleAccuracyMetric {
    constructor(params = {}) {
        this.name = 'example-accuracy';
        this.threshold = params.threshold !== undefined ? params.threshold : 0.5;
    }

    calculate(desired, actual) {
        return Math.abs(desired - actual) < this.threshold ? 1 : 0;
    }

    formulas() {
        return [`m = \\begin{cases} 1 & |d - y| < ${this.threshold} \\\\ 0 & \\text{otherwise} \\end{cases}`];
    }
}

// --- 11. Example Distribution ---
class ExampleUniformDistribution {
    constructor() {
        this.name = 'example-uniform';
        this.domain = 'bounded';
    }

    pdf(x) {
        return (x >= 0 && x <= 1) ? 1 : 0;
    }

    cdf(x) {
        if (x < 0) return 0;
        if (x > 1) return 1;
        return x;
    }

    sample(random = Math.random) {
        return random();
    }

    formulas(xExpr = 'x') {
        return [`f(${xExpr}) = \\begin{cases} 1 & 0 \\le ${xExpr} \\le 1 \\\\ 0 & \\text{otherwise} \\end{cases}`];
    }
    
    cumulativeFormulas(xExpr = 'x') {
        return [`F(${xExpr}) = \\begin{cases} 0 & ${xExpr} < 0 \\\\ ${xExpr} & 0 \\le ${xExpr} \\le 1 \\\\ 1 & ${xExpr} > 1 \\end{cases}`];
    }
}

// --- Registration ---
export function register(registry) {
    // Dynamically register the custom properties panel to the UI registry
    if (registry.registerPropertyView) {
        registry.registerPropertyView('example-ui-panel', ExamplePropertiesComponent);
    }

    registry.registerActivationFunction({
        name: 'example-step',
        functionClass: ExampleStepActivationFunction
    });

    registry.registerDerivativeActivationFunction({
        name: 'example-leaky',
        functionClass: ExampleLeakyActivationFunction
    });

    registry.registerLearningRule({
        name: 'example-learning-rule',
        ruleClass: ExampleLearningRule
    });

    registry.registerInitializer({
        name: 'example-initializer',
        label: 'Example Plugin Default',
        initializerClass: ExampleInitializer,
        paramsSchema: [
            { name: 'fillValue', label: 'Fill Value', type: 'number', defaultValue: 0.123 }
        ]
    });

    registry.registerErrorFunction({
        name: 'example-error-function',
        label: 'Example Huber Loss',
        functionClass: ExampleErrorFunction,
        paramsSchema: [
            { name: 'delta', label: 'Delta', type: 'number', defaultValue: 1.0, min: 0.1, max: 10.0, step: 0.1, description: 'Threshold between L1 and L2 penalty regions' }
        ]
    });

    registry.registerSynapse({
        type: 'example-synapse',
        synapseClass: ExampleSynapse
    });

    registry.registerModel({
        type: 'example-node',
        modelClass: ExampleModel
    });

    registry.registerNode({
        type: 'example-node',
        component: ExampleNodeComponent,
        modelClass: ExampleModel, // Attached for standard node creation (Flyweight)
        stateClass: ExampleNodeState, // Attached for standard state instantiation
        category: 'Plugins',
        color: '#f59e0b',
        validateConnection: (sourceType, targetType, sourceHandle, targetHandle) => {
            if (targetType === 'example-node') {
                return sourceType === 'input';
            }
            if (sourceType === 'example-node') {
                return targetType === 'output';
            }
            return false;
        }
    });

    registry.registerNormalization({
        name: 'example-winsorize',
        label: 'Example Winsorize (P5/P95 clamp)',
        normalizerClass: ExampleWinsorizeNormalization
    });

    registry.registerMetric({
        name: 'example-accuracy',
        label: 'Example Accuracy Metric',
        functionClass: ExampleAccuracyMetric,
        paramsSchema: [
            { name: 'threshold', label: 'Tolerance', type: 'number', defaultValue: 0.5, step: 0.01 }
        ]
    });

    registry.registerDistribution({
        name: 'example-uniform',
        label: 'Example Uniform Distribution',
        distributionClass: ExampleUniformDistribution
    });
}
