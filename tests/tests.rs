#[cfg(test)]
mod tests {
    use std::{collections::HashMap, time::{Duration, Instant}};

    use basic_graph::UnweightedGraph;
    use rand::Rng;
    #[test]
    #[should_panic]
    fn nonexisting_parent() {
        let mut graph = UnweightedGraph::from((
            1,
            vec![
                (vec![0], 2),    // 1
                (vec![0], 3),    // 2
                (vec![1], 4),    // 3
                (vec![1], 5),    // 4
                (vec![2], 6),    // 5
                (vec![2], 7),    // 6
            ],
        ));
        graph.add(vec![20],10);
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
        assert_eq!(vec![1,2,3,4,5,6,7,8,9],breadth_first);
    }
    #[test]
    fn application() {
        const w: usize = 100;
        const range: std::ops::Range<f32> = 1f32..10f32;
        const step: f32 = 0.001;
        const total_steps:f32 = (range.end-range.start)/step;

        let mut t_value = range.start;

        let initial_points = (0..2 * w)
            .map(|p| (function(t_value, p as f32) as usize, p))
            .collect::<Vec<_>>();
        // println!("initial_points: {:.3}->{:.?}", t_value,initial_points);

        let mut graph = UnweightedGraph::from(
            initial_points
                .into_iter()
                .map(|p| (p, range.start..range.start + step))
                .collect::<Vec<_>>(),
        );
        // println!("\ngraph: {}\n",graph);

        let mut prev_time = Duration::new(0,0);
        let mut interval_print = Instant::now();
        let mut total_calc_time = Instant::now();

        println!();
        t_value += step;
        while t_value <= range.end {
            // print!("\r{: <6}", format!("{:.2}", t_value));
            
            let start = Instant::now();
            let points = (0..2 * w)
                .map(|p| (function(t_value, p as f32) as usize, p))
                .collect::<Vec<_>>();
            prev_time += start.elapsed();
            

            graph
                .modify_insert_contiguous_set(
                    |((ai, aj), _), ((bi, bj), _)| ai == bi && aj == bj,
                    |(_, graph_range), (_, set_range)| range_add_assign(graph_range, set_range),
                    points
                        .into_iter()
                        .map(|p| (p, t_value..t_value + step))
                        .collect::<Vec<_>>(),
                );

            // println!("\ngraph: {}\n", graph);
            t_value += step;
            if interval_print.elapsed() > Duration::from_secs(5) {
                println!("{: <6} {:.3?}", format!("{:.2}", t_value),total_calc_time.elapsed());
                interval_print = Instant::now();
            }
        }
        println!();

        println!(
            "\nfinal_graph: {} Vs {} ({}kb)\n",
            graph.len(),
            ((range.end - range.start) / step) as usize,
            graph.len() * std::mem::size_of::<((usize, usize), std::ops::Range<f32>)>()
                / 1000
        );
        println!(
            "\t{}kb\n\t{}kb\n\t{}kb",
            graph.len() * std::mem::size_of::<((u16, u16), std::ops::Range<u16>)>() / 1000usize,
            graph.len() * std::mem::size_of::<((u16, u16), std::ops::Range<f32>)>() / 1000usize,
            graph.len() * std::mem::size_of::<((usize, usize), std::ops::Range<f32>)>() / 1000usize
        );


        let mut rng = rand::thread_rng();
        let mut space = (0..2*w+1).map(|_|(0..2*w+1).map(|_|{
            // rng.gen::<u8>()%100
            00
        }).collect::<Vec<_>>()).collect::<Vec<_>>();

        let line_to_fit = rng.gen_range(0..((range.end-range.start)/step) as usize);
        let angle_to_fit = line_to_fit as f32*step + range.start;
        println!("angle_to_fit: {}",angle_to_fit);
        for p in 0..2*w {
            let (i,j) = (function(angle_to_fit, p as f32) as usize, p);
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

        let mut space_map = space.into_iter().enumerate().flat_map(|(i,row)|{
            row.into_iter().enumerate().map(|(j,value)|((i,j),value)).collect::<Vec<_>>()
        }).collect::<HashMap<(usize,usize),u8>>();
        
        let mut cached_evaluations = HashMap::new();

        let start = Instant::now();
        let (_,interval) = graph.fold_filter_negative(
            ((0,0),range),
            |&x,b:&()|x>20, // Which is to say modify the accumulator on nodes where x>=20
            &(),
            &space_map,
            &mut cached_evaluations,
            |a,b|range_sub_assign(a,b),
            key_from_data
        );
        println!("interval: {:.?} {:.?} Vs {:.?}",interval,start.elapsed(),prev_time);

        fn key_from_data((a,_):&((usize,usize),std::ops::Range<f32>)) -> (usize,usize) {
            *a
        }
        fn function(t: f32, x: f32) -> f32 {
            let r = (w.pow(2) as f32 + t.powi(2)) / (2. * t);
            (r.powi(2) - (x - w as f32).powi(2)).sqrt() - (r.powi(2) - w.pow(2) as f32).sqrt()
        }
    }
    // Subtracts `other` from `this` such that `this` is modified to not overlap `other`.
    // Does not work when `other` covers `this` or vice-versa.
    fn range_sub_assign((_,mut this): ((usize,usize),std::ops::Range<f32>), (b,other): &((usize,usize),std::ops::Range<f32>)) -> ((usize,usize),std::ops::Range<f32>) {
        // print!(
        //     "\t sub: {:.?} {:.?}\t {} {} {}",
        //     this,other,other.end >= this.end && other.start < this.end && other.start > this.start,
        //     other.end > this.start && other.end < this.end && other.start <= this.start,
        //     other.end < this.end && other.start > this.start
        // );
        #[cfg(debug_assertions)]
        {
            if other.start < this.start && other.end > this.end {
                println!("\n\n");
                panic!("`other` ({:.?}) covers `this` ({:.?})",other,this);
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

        // // If `other` intersects upper part of `this`
        // if other.end > this.end && other.start < this.end && other.start > this.start {
        //     this.end = other.start;
            
        // }
        // // If `other` intersects lower part of `this`
        // else if other.end > this.start && other.end < this.end && other.start <= this.start {
        //     this.start = other.end;
        // }
        // // If `other` is covered by `this`
        // else if other.end < this.end && other.start > this.start {
        //     this.start = other.start;
        //     this.end = other.end;
        // }
            
        (*b,this)
    }
    // Adds two ranges together producing a new range which covers both original ranges.
    fn range_add_assign(this: &mut std::ops::Range<f32>, other: std::ops::Range<f32>) {
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
