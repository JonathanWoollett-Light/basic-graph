#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        time::{Duration, Instant},
    };

    use basic_graph::UnweightedGraph;
    use rand::Rng;
    #[test]
    #[should_panic]
    fn nonexisting_parent() {
        let mut graph = UnweightedGraph::from((
            1,
            vec![
                (vec![0], 2), // 1
                (vec![0], 3), // 2
                (vec![1], 4), // 3
                (vec![1], 5), // 4
                (vec![2], 6), // 5
                (vec![2], 7), // 6
            ],
        ));
        graph.add(vec![20], 10);
    }
    #[test]
    fn iter() {
        // 1 -> {2,3}
        // 2 -> {4,5}
        // 3 -> {6,7}
        // {4,6} -> { 8 }
        // {5,7} -> { 9 }
        //
        // 1 ─┬─ 2 ─┬─ 4 ───┬─ 8
        //    │     └─ 5 ━┳━┿━ 9
        //    └─ 3 ─┬─ 6 ─╂─┘
        //          └─ 7 ━┛
        let graph = UnweightedGraph::from((
            1,
            vec![
                (vec![0], 2),    // 1
                (vec![0], 3),    // 2
                (vec![1], 4),    // 3
                (vec![1], 5),    // 4
                (vec![2], 6),    // 5
                (vec![2], 7),    // 6
                (vec![3, 5], 8), // 7
                (vec![4, 6], 9), // 8
            ],
        ));
        let breadth_first = graph.iter().cloned().collect::<Vec<_>>();
        assert_eq!(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], breadth_first);
    }
    // Consider a problem, we are searching for a line like `y=f(x,a,b,...)` in a MxN space
    //  (whereh `a,b,...` are constants).
    // Alternatively to simply brute force searching, we can construct a graph of the size of the
    //  space MxN, where each node contains the combinations of the parameters (a,b,...) for each
    //  line which passes through the cell this node is respective to.
    // Then after caching this graph (it will be a DAG), to find a line we can simply breadth-first
    //  search fold through our graph, starting with the full range, we can then progressively
    //  subtract possible values at each leaf node, leaving us with the combinations of paramters
    //  for all lines which fit.
    // In the specific case where we have 1 parameter we can swap storing combinations of
    //  paramters in our nodes for a range of this parameter, this allows us to search
    //  faster and more accurately.
    #[test]
    fn application() {
        // The width of the space
        const W: usize = 500;
        // The range of possible values we consider for the parameters forming our line.
        const VALUE_RANGE: std::ops::Range<f32> = 1f32..10f32;
        // When constructing our graph we brute force through equation, this determines the
        //  interval between the paramter values (this should only affect intiial graph
        //  construction speed and deterines the accuracy of our graph).
        const STEP: f32 = 0.01;
        println!(
            "total steps: {}",
            ((VALUE_RANGE.end - VALUE_RANGE.start) / STEP) as u32
        );

        let mut t_value = VALUE_RANGE.start;

        let initial_points = (0..2 * W)
            .map(|p| (curve_function(t_value, p as f32) as usize, p))
            .collect::<Vec<_>>();
        // println!("initial_points: {:.3}->{:.?}", t_value,initial_points);

        let mut graph = UnweightedGraph::from(
            initial_points
                .into_iter()
                .map(|p| (p, VALUE_RANGE.start..VALUE_RANGE.start + STEP))
                .collect::<Vec<_>>(),
        );
        // println!("\ngraph: {}\n",graph);

        let mut prev_time = Duration::new(0, 0);
        // let mut interval_print = Instant::now();
        // let total_calc_time = Instant::now();

        t_value += STEP;
        // println!("{: <6}",format!("{:.2}", VALUE_RANGE.end));
        while t_value <= VALUE_RANGE.end {
            // print!("\r{: <6}", format!("{:.2}", t_value));

            let start = Instant::now();
            let points = (0..2 * W)
                .map(|p| (curve_function(t_value, p as f32) as usize, p))
                .collect::<Vec<_>>();
            prev_time += start.elapsed();

            graph.modify_insert_contiguous_set(
                |((ai, aj), _), ((bi, bj), _)| ai == bi && aj == bj,
                |(_, graph_range), (_, set_range)| cover_range(graph_range, set_range),
                points
                    .into_iter()
                    .map(|p| (p, t_value..t_value + STEP))
                    .collect::<Vec<_>>(),
            );

            // println!("\ngraph: {}\n", graph);
            t_value += STEP;
            // if interval_print.elapsed() > Duration::from_secs(1) {
            //     print!(
            //         "\r{: <6} {:?}    ",
            //         format!("{:.2}", t_value),
            //         total_calc_time.elapsed()
            //     );
            //     interval_print = Instant::now();
            // }
        }
        // println!();

        println!(
            "final_graph: {} Vs {} ({}kb)",
            ((VALUE_RANGE.end - VALUE_RANGE.start) / STEP) as usize,
            graph.len(),
            graph.len() * std::mem::size_of::<((usize, usize), std::ops::Range<f32>)>() / 1000
        );
        println!(
            "{}kb\t{}kb\t{}kb",
            graph.len() * std::mem::size_of::<((u16, u16), std::ops::Range<u16>)>() / 1000usize,
            graph.len() * std::mem::size_of::<((u16, u16), std::ops::Range<f32>)>() / 1000usize,
            graph.len() * std::mem::size_of::<((usize, usize), std::ops::Range<f32>)>() / 1000usize
        );

        let mut rng = rand::thread_rng();
        let mut space = (0..2 * W + 1)
            .map(|_| {
                (0..2 * W + 1)
                    .map(|_| {
                        // rng.gen::<u8>()%100
                        00
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let line_to_fit = rng.gen_range(0..((VALUE_RANGE.end - VALUE_RANGE.start) / STEP) as usize);
        let angle_to_fit = line_to_fit as f32 * STEP + VALUE_RANGE.start;
        println!("angle_to_fit: {}", angle_to_fit);
        for p in 0..2 * W {
            let (i, j) = (curve_function(angle_to_fit, p as f32) as usize, p);
            space[i][j] = 99;
        }

        // println!("space:");
        // for r in space.iter() {
        //     for v in r.iter() {
        //         print!("{:02} ",v);
        //     }
        //     println!();
        // }
        // for r in space.iter() {
        //     for v in r.iter() {
        //         print!("{} ",if *v<20 { "X" } else { "*" });
        //     }
        //     println!();
        // }

        // The hashmap of the coordinates of the space mapped to the values at the respective cell at the coordinates.
        let space_map = space
            .into_iter()
            .enumerate()
            .flat_map(|(i, row)| {
                row.into_iter()
                    .enumerate()
                    .map(|(j, value)| ((i, j), value))
                    .collect::<Vec<_>>()
            })
            .collect::<HashMap<(usize, usize), u8>>();

        let start = Instant::now();
        let (_, interval) = graph.fold_filter_negative(
            ((0, 0), VALUE_RANGE),
            // Modify the accumulator on nodes where x>20.
            // Which is to say, search for a line where all cells it passes through have a value less than `<=20`.
            |&x| x > 20,
            // The hashmap of coordinates to cell values.
            &space_map,
            // Modifiy accumulator such that it is disjoint with the ranges in nodes which don't satify `|&x| x > 20`.
            |a, b| disjoin(a, b),
            // Extract coordinates from stored data
            |(a,_)|*a,
        );
        println!(
            "interval: {:.?} {:.?} Vs {:.?}",
            interval,
            start.elapsed(),
            prev_time
        );

        const ALLOWED: f32 = 10. * STEP;
        println!("ALLOWED: {}",ALLOWED);
        assert!(interval.start - ALLOWED < angle_to_fit && interval.end + ALLOWED > angle_to_fit,"!({:?}).contains({})",(interval.start - ALLOWED)..(interval.end + ALLOWED),angle_to_fit);

        // `t`: Curve constant of line, `x`: x coordinate.
        fn curve_function(t: f32, x: f32) -> f32 {
            let r = (W.pow(2) as f32 + t.powi(2)) / (2. * t);
            (r.powi(2) - (x - W as f32).powi(2)).sqrt() - (r.powi(2) - W.pow(2) as f32).sqrt()
        }
    }
    // Modifies `this` to be disjoint from `other`.
    // Does not work when `other` covers `this` or vice-versa.
    fn disjoin(
        (_, mut this): ((usize, usize), std::ops::Range<f32>),
        (b, other): &((usize, usize), std::ops::Range<f32>),
    ) -> ((usize, usize), std::ops::Range<f32>) {
        #[cfg(debug_assertions)]
        {
            if other.start < this.start && other.end > this.end {
                println!("\n\n");
                panic!("`other` ({:.?}) covers `this` ({:.?})", other, this);
            }
        }

        // If `other` intersects upper part of `this`
        if other.end > this.start && other.end < this.end {
            this.start = other.end
        }
        // If `other` intersects lower part of `this`
        else if other.start > this.start && other.start < this.end {
            this.end = other.start
        }

        (*b, this)
    }
    // Modifies `this` to extend it to cover `other`.
    fn cover_range(this: &mut std::ops::Range<f32>, other: std::ops::Range<f32>) {
        *this = std::ops::Range {
            start: if this.start < other.start {
                this.start
            } else {
                other.start
            },
            end: if this.end > other.end {
                this.end
            } else {
                other.end
            },
        }
    }
}
