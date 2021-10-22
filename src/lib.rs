use core::{fmt, fmt::Debug};
use std::{
    collections::{HashMap, VecDeque},
    hash::Hash,
};

/// Unweighted, directed graph (a tree where branches can join).
///
/// Designed to be used for caching constant information for later use.
#[derive(Debug)]
pub struct UnweightedGraph<T> {
    nodes: Vec<Node<T>>,
}
impl<T: Debug> UnweightedGraph<T> {
    pub fn new(x: T) -> Self {
        UnweightedGraph {
            nodes: vec![Node {
                children: Vec::new(),
                data: x,
            }],
        }
    }
    /// Adds a node to the graph.
    ///
    /// Every element in `parents` must correspond to an index in the underlying `vec<Node<T>>` (which stores nodes in the order they where inserted).
    /// ```
    /// // 1 -> {2,3}
    /// // 2 -> {4,5}
    /// // 3 -> {6,7}
    /// // {4,6} -> { 8 }
    /// // {5,7} -> { 9 }
    /// //
    /// // 1 ─┬─ 2 ─┬─ 4 ───┬─ 8
    /// //    │     └─ 5 ━┳━┿━ 9
    /// //    └─ 3 ─┬─ 6 ─╂─┘
    /// //          └─ 7 ━┛
    /// let mut graph = basic_graph::UnweightedGraph::new(1);
    /// graph.add(vec![0],2);
    /// graph.add(vec![0],3);
    /// graph.add(vec![1],4);
    /// graph.add(vec![1],5);
    /// graph.add(vec![2],6);
    /// graph.add(vec![2],7);
    /// graph.add(vec![3,5],8);
    /// graph.add(vec![4,6],9);
    /// let breadth_first = graph.iter().cloned().collect::<Vec<_>>();
    /// assert_eq!(vec![1,2,3,4,5,6,7,8,9],breadth_first);
    /// ```
    pub fn add(&mut self, parents: Vec<usize>, data: T) -> usize {
        let len = self.nodes.len();
        for parent in parents {
            self.nodes[parent].children.push(len);
        }
        self.nodes.push(Node::new(data));
        len
    }
    /// Inserts a set of consecutively linked elements into the graph, modifying existing matching elements if found to already be present.
    ///
    /// More precisely, each element at index `n` in `contiguous_set` is considered the parent of the element at index `n+1` in `contiguous_set`.
    ///
    /// The 1st element in `contiguous_set` must link to an existing element.
    ///
    /// `O(k * (n+k)/2)`: `k=contiguous_set.len()`, `n=self.len()`
    ///
    /// Since the insertion of each node in the given set may require searching the entire graph, we have `k` insertions of Onation `n, n+1, n+2, ..., n+k` averaging to `(n+k)/2` thus `k*(n+k)/2)`.
    pub fn modify_insert_contiguous_set(
        &mut self,
        find: fn(&T, &T) -> bool,
        modify: fn(&mut T, T),
        contiguous_set: Vec<T>,
    ) {
        // imho this is nicer than wrapping whole functionality inside `if let Some(first) = iter.next() { ... }`.
        if contiguous_set.is_empty() {
            return;
        }
        let mut iter = contiguous_set.into_iter();
        let first = iter.next().unwrap();
        let mut set_parent_index = self
            .nodes
            .iter()
            .enumerate()
            .find_map(|(index, n)| {
                if find(&n.data, &first) {
                    Some(index)
                } else {
                    None
                }
            })
            .expect("1st element in set doesn't link to existing element.");

        for value in iter {
            // Search for node under parent.
            let found_index_parent_opt =
                self.nodes[set_parent_index]
                    .children
                    .iter()
                    .find_map(|&index| {
                        if find(&self.nodes[index].data, &value) {
                            Some(index)
                        } else {
                            None
                        }
                    });
            // If we found the node as a child of its parent.
            set_parent_index = if let Some(found_index) = found_index_parent_opt {
                // Modify node
                modify(&mut self.nodes[found_index].data, value);
                found_index
            // Else if we did not find the node under its parent
            } else {
                // Search for the node in the whole graph
                let found_index_opt = self.nodes.iter().enumerate().find_map(|(index, node)| {
                    if find(&node.data, &value) {
                        Some(index)
                    } else {
                        None
                    }
                });
                // If we found node somewhere in the graph
                if let Some(found_index) = found_index_opt {
                    // Modify node
                    modify(&mut self.nodes[found_index].data, value);
                    // Set as child of the parent from the set
                    self.nodes[set_parent_index].children.push(found_index);
                    found_index
                }
                // If couldn't find the node in the graph.
                else {
                    // We add the node to the graph, as a child of its parent from the set.
                    self.add(vec![set_parent_index], value)
                }
            }
        }
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}
impl<T: Clone> UnweightedGraph<T> {
    /// Starting at `self.nodes[0]` with `self.nodes[0].data` folds the `T` value through the graph via `modify` selecting the 1st children of each parent
    /// for which `find` returns true, given the respective value from `set`, returning when no child of the current node causes `find` to be true.
    ///
    /// It must be possible to fold through the given set with this graph.
    pub fn fold_through_set<P>(
        &self,
        set: Vec<P>,
        find: fn(&T, &P) -> bool,
        modify: fn(T, &T) -> T,
    ) -> T {
        let mut value = self.nodes[0].data.clone();
        let cur = &self.nodes[0];
        for set_value in set.into_iter() {
            let next = cur
                .children
                .iter()
                .find(|&&child_index| find(&self.nodes[child_index].data, &set_value))
                .unwrap();
            value = modify(value, &self.nodes[*next].data);
        }
        value
    }
}
impl<T: Clone + Debug> UnweightedGraph<T> {
    /// Folds an accumulator through the graph in breadth first order, applying the modification only on nodes for which the evaluation function `eval` returns false, and only exploring children of nodes for which the evaluation function `eval` returns true.
    ///
    /// With a bunch of adjustments to allow for using external evaluation data and caching these evaluations between usages.
    ///
    /// For every node `index` this must be true: `eval_data.get(key_from_data(self.nodes[index].data)).is_some || eval_map.get(key_from_data(&self.nodes[index].data)).is_some`.
    pub fn fold_filter_negative<'a, Key: Eq + Hash, Data: Debug, EvalParameters>(
        &'a self,
        // Accumulator like typical `fold`
        mut acc: T,
        // Evaluates node per respective evaluation data
        eval: fn(&Data, &EvalParameters) -> bool,
        // Any additional evlauation paramters.
        eval_parameters: &EvalParameters,
        // Link graph data to evaluation data
        eval_data: &HashMap<Key, Data>,
        // Caching results of evaluation
        eval_map: &mut HashMap<Key, bool>,
        // typical `fold` function
        modify: fn(T, &T) -> T,
        key_from_data: fn(&T) -> Key,
    ) -> T {
        let mut visited = vec![false; self.nodes.len()];
        let mut queue = VecDeque::new();

        if UnweightedGraph::evaluation(
            &self.nodes[0].data,
            eval,
            eval_parameters,
            eval_data,
            eval_map,
            key_from_data,
        ) {
            visited[0] = true;
            queue.push_back(0);
            while let Some(next) = queue.pop_front() {
                // println!("{}:",next);

                for &child in self.nodes[next].children.iter() {
                    if !visited[child] {
                        visited[child] = true;

                        // print!("\t{}:({:.?}) | {:.?}",child,self.nodes[child].data,acc);

                        if UnweightedGraph::evaluation(
                            &self.nodes[child].data,
                            eval,
                            eval_parameters,
                            eval_data,
                            eval_map,
                            key_from_data,
                        ) {
                            queue.push_back(child);

                            // println!();
                        } else {
                            // We modify the accumulator based on the first negative children of a path through the graph
                            // E.g. If we have a graph of ((x,y),a..b) and want to find the subset range x..z of a..b where
                            //  all (x,y)s are true, we want to progressively exclude the ranges a_n..b_n where (x_n,y_n)
                            //  are false.
                            // In our specific application if all parent of a node are false, then the
                            //  child is false, and furthermore this means the children of this child
                            //  (which only have this child as their parent) have ranges which are a
                            //  subset of this child's range and will not restrict the overall range.
                            //  Thus if a node is negative we need not explore its children, if
                            //   some of its children are useful to explore there will be a positive
                            //   node which links to them. Thus can explore these useful children
                            //   through that node.
                            acc = modify(acc, &self.nodes[child].data);

                            // println!("->{:.?}",acc);
                        }
                    }
                }
            }
        } else {
            acc = modify(acc, &self.nodes[0].data);
        }
        return acc;
    }
    fn evaluation<'a, Key: Eq + Hash, Data: Debug, EvalParameters>(
        data: &'a T,
        eval: fn(&Data, &EvalParameters) -> bool,
        eval_parameters: &EvalParameters,
        eval_data: &HashMap<Key, Data>,
        eval_map: &mut HashMap<Key, bool>,
        key_from_data: fn(&T) -> Key,
    ) -> bool {
        let key = key_from_data(data);
        match eval_map.get(&key) {
            Some(&cached) => cached,
            None => {
                let data = eval_data.get(&key).unwrap();

                // print!(" {{{:.?}}}",data);

                let temp_eval = eval(data, eval_parameters);
                eval_map.insert(key, temp_eval);
                temp_eval
            }
        }
    }
}
/// Adds many nodes consecutively to a graph.
///
/// Simply calls [`add`](UnweightedGraph<T>::add) on all elements.
///
/// Every element in `parents` must correspond to an index in the underlying `vec<Node<T>>` (which stores nodes in the order they where inserted).
/// ```
/// // 1 -> {2,3}
/// // 2 -> {4,5}
/// // 3 -> {6,7}
/// // {4,6} -> { 8 }
/// // {5,7} -> { 9 }
/// //
/// // 1 ─┬─ 2 ─┬─ 4 ───┬─ 8
/// //    │     └─ 5 ━┳━┿━ 9
/// //    └─ 3 ─┬─ 6 ─╂─┘
/// //          └─ 7 ━┛
/// let graph = basic_graph::UnweightedGraph::from((
///     1,                  // 0
///     vec![
///         (vec![0], 2),   // 1
///         (vec![0], 3),   // 2
///         (vec![1], 4),   // 3
///         (vec![1], 5),   // 4
///         (vec![2], 6),   // 5
///         (vec![2], 7),   // 6
///         (vec![3, 5], 8),// 7
///         (vec![4, 6], 9),// 8
///     ],
/// ));
/// let breadth_first = graph.iter().cloned().collect::<Vec<_>>();
/// assert_eq!(vec![1,2,3,4,5,6,7,8,9],breadth_first);
/// ```
impl<T: Debug> From<(T, Vec<(Vec<usize>, T)>)> for UnweightedGraph<T> {
    fn from((head, dataset): (T, Vec<(Vec<usize>, T)>)) -> Self {
        let mut graph = UnweightedGraph::new(head);
        for (parents, data) in dataset.into_iter() {
            graph.add(parents, data);
        }
        graph
    }
}
impl<'a, T: Debug> UnweightedGraph<T> {
    /// Breadth first iterator.
    pub fn iter(&'a self) -> UnweightedGraphCounter<'a, T> {
        UnweightedGraphCounter::new(&self)
    }
}
impl<T: Debug> fmt::Display for UnweightedGraph<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let string = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| format!("\n{}:{:.?}->{:?}", i, n.data, n.children))
            .collect::<String>();
        write!(f, "{}", string)
    }
}
/// Inserts a set of consecutively linked elements into the graph.
///
/// Like [`modify_insert_contiguous_set`](UnweightedGraph<T>::modify_insert_contiguous_set) but without modifying.
///
/// And without checking if the parent of the 1st node exists, thus this can effectively split the graph.
impl<T: Debug> From<Vec<T>> for UnweightedGraph<T> {
    fn from(mut vec: Vec<T>) -> Self {
        let last_opt = vec.pop();
        let mut nodes = vec
            .into_iter()
            .enumerate()
            .map(|(index, data)| Node {
                children: vec![index + 1],
                data,
            })
            .collect::<Vec<_>>();
        if let Some(last) = last_opt {
            nodes.push(Node::new(last));
        }
        UnweightedGraph { nodes }
    }
}
/// Iteration struct for `UnweightedGraph`
pub struct UnweightedGraphCounter<'a, T: Debug> {
    data: &'a UnweightedGraph<T>,
    visited: Vec<bool>,
    queue: VecDeque<usize>,
}
impl<'a, T: Debug> UnweightedGraphCounter<'a, T> {
    pub fn new(data: &'a UnweightedGraph<T>) -> Self {
        Self {
            data,
            visited: vec![false; data.nodes.len()],
            queue: VecDeque::new(),
        }
    }
}
impl<'a, T: Debug> Iterator for UnweightedGraphCounter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        // On first iteration
        if !self.visited[0] {
            self.visited[0] = true;
            self.queue.push_back(0);
        }
        if let Some(next) = self.queue.pop_front() {
            for &index in self.data.nodes[next].children.iter() {
                // If children not visited
                if !self.visited[index] {
                    self.visited[index] = true;
                    self.queue.push_back(index);
                }
            }
            Some(&self.data.nodes[next].data)
        } else {
            None
        }
    }
    /// O(n)
    fn size_hint(&self) -> (usize, Option<usize>) {
        // The number of remains element is equal to the count of the number of element which are `false` visited, not yet visited
        let remaining = self.visited.iter().filter(|&v| !v).count();
        (remaining, Some(remaining))
    }
}
impl<'a, T: Debug> ExactSizeIterator for UnweightedGraphCounter<'a, T> {
    fn len(&self) -> usize {
        self.data.nodes.len()
    }
}
#[derive(Debug)]
pub struct Node<T> {
    children: Vec<usize>,
    data: T,
}
impl<T: Debug> Node<T> {
    pub fn new(data: T) -> Self {
        Node {
            children: Vec::new(),
            data,
        }
    }
}
