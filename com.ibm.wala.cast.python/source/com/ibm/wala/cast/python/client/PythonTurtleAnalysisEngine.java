package com.ibm.wala.cast.python.client;

import java.io.IOException;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ipa.summaries.TurtleSummary;
import com.ibm.wala.cast.tree.CAstSourcePositionMap.Position;
import com.ibm.wala.cast.util.SourceBuffer;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.ipa.callgraph.AnalysisOptions;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAGetInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.types.MemberReference;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.collections.HashSetFactory;

public class PythonTurtleAnalysisEngine extends PythonAnalysisEngine<Set<PythonTurtleAnalysisEngine.TurtlePath>> {

	private TurtleSummary turtles;
	
	@Override
	protected void addBypassLogic(AnalysisOptions options) {
		super.addBypassLogic(options);
		addSummaryBypassLogic(options, "numpy_turtle.xml");
		
		turtles = new TurtleSummary(getClassHierarchy());
		
		turtles.analyzeWithTurtles(options);
	}

	private List<MemberReference> makePath(CallGraph CG, CGNode node, DefUse du, int vn) {
		SSAInstruction def = du.getDef(vn);
		if (def instanceof SSAAbstractInvokeInstruction) {
			if (((SSAAbstractInvokeInstruction)def).getDeclaredTarget().getName().toString().equals("import")) {
				return Collections.singletonList(((SSAAbstractInvokeInstruction)def).getDeclaredTarget());
			} else if (CG.getPossibleTargets(node, ((SSAAbstractInvokeInstruction)def).getCallSite()).toString().contains("turtle")) {
				return makePath(CG, node, du, ((SSAAbstractInvokeInstruction)def).getReceiver());
			}
		} else if (def instanceof SSAGetInstruction) {
			List<MemberReference> stuff = new LinkedList<>(makePath(CG, node, du, ((SSAGetInstruction)def).getRef()));
			stuff.add(0, ((SSAGetInstruction)def).getDeclaredField());
			return stuff;
		} 
		
		return Collections.emptyList();
	}
	
	public static interface TurtlePath {
		PointerKey value();
		List<MemberReference> path();
		Position position();
		
		default boolean hasSuffix(List<MemberReference> suffix) {
			List<MemberReference> path = path();
			if (suffix.size() > path.size()) {
				return false;
			} else {
				int d = path.size() - suffix.size();
				for(int i = suffix.size()-1; i >= 0; i--) {
					if (! (suffix.get(i).equals(path.get(i+d)))) {
						return false;
					}
				}
				
				return true;
			}
		}
	}
	
	@Override
	public Set<TurtlePath> performAnalysis(PropagationCallGraphBuilder builder) throws CancelException {
		Set<TurtlePath> turtlePaths = HashSetFactory.make();
		CallGraph CG = builder.getCallGraph();
		CG.getNodes(turtles.getCode().getReference()).forEach((CGNode turtle) -> {
			CG.getPredNodes(turtle).forEachRemaining((CGNode caller) -> {
				IR callerIR = caller.getIR();
				DefUse DU = caller.getDU();
				CG.getPossibleSites(caller, turtle).forEachRemaining((CallSiteReference site) -> {
					 for(SSAAbstractInvokeInstruction inst : callerIR.getCalls(site)) {
						 turtlePaths.add(new TurtlePath() {
							private final List<MemberReference> path = makePath(CG, caller, DU, inst.getDef());
							
							@Override
							public PointerKey value() {
								return builder.getPointerKeyForLocal(caller, inst.getDef());
							}

							@Override
							public List<MemberReference> path() {
								return path;
							}

							@Override
							public Position position() {
								return ((AstMethod)callerIR.getMethod()).debugInfo().getInstructionPosition(inst.iindex);
							}
							 
							@Override
							public String toString() {
								StringBuffer out = new StringBuffer();
								try {
									out.append(new SourceBuffer(((AstMethod)callerIR.getMethod()).debugInfo().getInstructionPosition(inst.iindex)));
								} catch (IOException e) {
									out.append("v").append(inst.getDef());
								}
								out.append(":");
								out.append(path());
								for(int i = 1; i < inst.getNumberOfUses(); i++) {
									List<MemberReference> path = makePath(CG, caller, DU, inst.getUse(i));
									if (! path.isEmpty()) {
										try {
											out.append("\n  ").append(new SourceBuffer(((AstMethod)callerIR.getMethod()).debugInfo().getOperandPosition(inst.iindex, i)));
										} catch (IOException e) {
											out.append("\n  arg ").append(i);
										}
										out.append(":").append(path());
									 }
								}
							 	return out.toString();
							}
						});
					 }
				});
			});
		});
		return turtlePaths;
	}
}