"""Cover all activation radii when expanding the existing alpha search."""
import ast
import hashlib
import inspect
import time
from . import common as c


def main():
    from . import joint_mse_search_002 as original
    source = inspect.getsource(original.main)
    tree = ast.parse(source)

    class Domain(ast.NodeTransformer):
        replacements = 0

        def visit_Constant(self, node):
            if isinstance(node.value, (int, float)) and node.value == 4:
                self.replacements += 1
                return ast.copy_location(ast.Name(id="upper", ctx=ast.Load()), node)
            if node.value == "JOINT_MSE_DIAGNOSTIC_002.json":
                return ast.copy_location(ast.Constant("EXTENDED_RADIUS_PROPOSAL_005.json"), node)
            if node.value == "mse_search_progress_002.json":
                return ast.copy_location(ast.Constant("extended_radius_progress_005.json"), node)
            return node

    transform = Domain()
    tree = transform.visit(tree)
    assert transform.replacements == 5, "All five domain endpoints/cut filters must expand together"
    function = tree.body[0]
    index = next(i for i, node in enumerate(function.body) if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "ratio")
    definition = ast.parse("upper = float(max(4., ratio[np.isfinite(ratio)].max()))").body[0]
    function.body.insert(index+1, definition)
    ast.fix_missing_locations(tree)
    changed = ast.unparse(tree) + "\n"
    # The prior draft expanded endpoints but omitted one integer-valued cut
    # filter. Keep it as a superseded proposal draft; no score used it.
    c.save(c.HERE / "EXTENDED_RADIUS_ANALYSIS_SOURCE_005.json", dict(at=time.time(),
        original_source_sha256=hashlib.sha256(source.encode()).hexdigest(), transformed_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        expanded_domain_constants=transform.replacements, transformed_source=changed,
        supersedes="extended_radius_analysis_003.py", reason="Expand the activation-cut filter as well as domain endpoints; proposal only, no official score or frozen grid changed.",
        P_grid_sha256=c.digest(c.HERE / "GRID.json"), neural_forwards=0))
    namespace = dict(vars(original))
    exec(compile(changed, "<P_extended_alpha_all_activation_cuts_004>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
