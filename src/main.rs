use std::{collections::VecDeque, ops::Range};

use petgraph::prelude::UnGraph;
use piston_window::{PistonWindow, WindowSettings};
use plotters::prelude::*;
use plotters_piston::{draw_piston_window, PistonBackend};
use rand::Rng;

fn sample_range_with_resampling(points: &VecDeque<(i32, i32)>, range: Range<i32>) -> (i32, i32) {
    let mut rng = rand::thread_rng();
    loop {
        let random_x = rng.gen_range(range.clone());
        let random_y = rng.gen_range(range.clone());
        if !points.contains(&(random_x, random_y)) {
            return (random_x, random_y);
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct NodeData {
    x: i32,
    y: i32,
}
#[derive(Debug, Copy, Clone)]
struct EdgeData {
    dist: f64,
}

fn is_fully_connected(graph: &UnGraph<NodeData, EdgeData>) -> bool {
    let nodes = graph.node_indices();
    for node in nodes.clone() {
        let count = graph.neighbors(node).count();
        if count != nodes.clone().count() - 1 {
            return false;
        }
    }
    true
}

fn initialize_graph(graph: &mut UnGraph<NodeData, EdgeData>, points: &VecDeque<(i32, i32)>) {
    let mut node_map = std::collections::HashMap::new();
    for (x, y) in points.iter() {
        let node = graph.add_node(NodeData { x: *x, y: *y });
        node_map.insert((x, y), node);
    }

    for (i, (x1, y1)) in points.iter().enumerate() {
        for (x2, y2) in points.iter().skip(i + 1) {
            if (x1, y1) != (x2, y2) {
                let dist = ((x1 - x2).pow(2) + (y1 - y2).pow(2)) as f64;
                graph.add_edge(
                    *node_map.get(&(x1, y1)).unwrap(),
                    *node_map.get(&(x2, y2)).unwrap(),
                    EdgeData { dist },
                );
            }
        }
    }
    assert!(is_fully_connected(graph)); // making sure that the graph is fully connected
}

const FPS: u32 = 10;
const MAX_POINTS: usize = 30;
const WINDOW_TITLE: &str = "Traveling Salesman Problem";

enum State {
    Populating,
    MakingPath,
}

fn main() {
    let mut window: PistonWindow = WindowSettings::new(WINDOW_TITLE, [1100, 1100])
        .samples(4)
        .build()
        .unwrap();
    let mut points = VecDeque::new();
    let mut graph = petgraph::Graph::new_undirected();
    let mut state = State::Populating;
    let mut f = |b: PistonBackend| {
        let root = b.into_drawing_area();
        root.fill(&WHITE)?;

        let mut cc = ChartBuilder::on(&root)
            .set_label_area_size(LabelAreaPosition::Left, 40)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .caption(WINDOW_TITLE, ("sans-serif", 40))
            .build_cartesian_2d(-500..501, -500..501)?;

        cc.configure_mesh().draw()?;

        match &state {
            State::Populating => {
                let range = -500..501;
                let coord = sample_range_with_resampling(&points, range);
                points.push_front(coord);
                if points.len() == MAX_POINTS {
                    state = State::MakingPath;
                    initialize_graph(&mut graph, &points);
                }
            }
            State::MakingPath => {}
        }

        cc.draw_series(points.iter().map(|(x, y)| Circle::new((*x, *y), 15, RED)))
            .unwrap();

        std::thread::sleep(std::time::Duration::from_millis(1000 / FPS as u64));
        Ok(())
    };
    while draw_piston_window(&mut window, &mut f).is_some() {}
}
