import torch
import torch.fx
from . import ops


class AutoSoftener:
    """
    自动软化层：对标记为离散的节点进行软化替换。
    """
    
    def __init__(self, eps=1e-3):
        self.eps = eps
    
    def _replace_with_threshold(self, graph, node):
        """替换为阈值软化"""
        with graph.inserting_after(node):
            new_node = graph.call_function(
                ops.heaviside,
                args=(node.args[0], self.eps),
                kwargs={}
            )
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        return new_node
    
    def _replace_with_round(self, graph, node):
        """替换为四舍五入软化"""
        with graph.inserting_after(node):
            new_node = graph.call_function(
                ops.round,
                args=(node.args[0], self.eps),
                kwargs={}
            )
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        return new_node
    
    def _replace_with_sign(self, graph, node):
        """替换为符号函数软化"""
        with graph.inserting_after(node):
            new_node = graph.call_function(
                ops.sign,
                args=(node.args[0], self.eps),
                kwargs={}
            )
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        return new_node
    
    def _replace_with_floor(self, graph, node):
        """替换为向下取整软化"""
        with graph.inserting_after(node):
            new_node = graph.call_function(
                ops.floor,
                args=(node.args[0], self.eps),
                kwargs={}
            )
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        return new_node
    
    def _replace_with_ceil(self, graph, node):
        """替换为向上取整软化"""
        with graph.inserting_after(node):
            new_node = graph.call_function(
                ops.ceil,
                args=(node.args[0], self.eps),
                kwargs={}
            )
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        return new_node
    
    def _replace_with_argmax(self, graph, node):
        """替换为 argmax 软化"""
        with graph.inserting_after(node):
            new_node = graph.call_function(
                ops.argmax,
                args=(node.args[0],),
                kwargs={'dim': node.kwargs.get('dim', -1), 'eps': self.eps}
            )
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        return new_node
    
    def _replace_with_detach(self, graph, node):
        """处理 detach 操作：用 soft_detach 替代"""
        with graph.inserting_after(node):
            new_node = graph.call_function(
                self._soft_detach,
                args=(node.args[0],),
                kwargs={'eps': self.eps}
            )
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        return new_node
    
    def _soft_detach(self, x, eps=1e-3):
        """软 detach：保留微小梯度"""
        return x - eps * x.detach() + eps * x
    
    def _select_strategy(self, node):
        """根据节点类型和离散类型选择软化策略"""
        discrete_type = node.meta.get('discrete_type')
        target_name = str(node.target)
        
        if target_name in ('torch.sign', 'sign'):
            return self._replace_with_sign
        elif target_name in ('torch.round', 'round'):
            return self._replace_with_round
        elif target_name in ('torch.floor', 'floor'):
            return self._replace_with_floor
        elif target_name in ('torch.ceil', 'ceil'):
            return self._replace_with_ceil
        elif target_name in ('torch.argmax', 'argmax'):
            return self._replace_with_argmax
        elif target_name in ('torch.detach', 'detach'):
            return self._replace_with_detach
        
        if discrete_type == 'type_transition':
            return self._replace_with_threshold
        elif discrete_type == 'numerical_jump':
            return self._replace_with_sign
        elif discrete_type == 'gradient_blackhole':
            return self._replace_with_detach
        elif discrete_type == 'output_clustering':
            return self._replace_with_threshold
        
        return self._replace_with_threshold
    
    def soften_graph(self, fx_module, skip_set=None):
        """
        对 FX 图中的离散节点进行软化替换。
        
        参数:
            fx_module: torch.fx.GraphModule
            skip_set: 黑名单，包含不需要软化的节点名称
        
        返回:
            软化后的 fx_module
        """
        if skip_set is None:
            skip_set = set()
        
        graph = fx_module.graph
        nodes_to_soften = []
        
        for node in graph.nodes:
            if node.meta.get('is_discrete', False):
                if node.name in skip_set:
                    continue
                nodes_to_soften.append(node)
        
        for node in nodes_to_soften:
            strategy = self._select_strategy(node)
            strategy(graph, node)
        
        graph.lint()
        fx_module.recompile()
        
        return fx_module
