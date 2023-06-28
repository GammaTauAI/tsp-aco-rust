use std::{collections::VecDeque, ops::Range};

use petgraph::{graph::NodeIndex, prelude::UnGraph, Graph};
use piston_window::{PistonWindow, WindowSettings};
use plotters::prelude::*;
use plotters_piston::{draw_piston_window, PistonBackend};
use rand::Rng;

/// i know, i know, this is a terrible way to do this
fn sample_range_with_resampling(points: &VecDeque<(i32, i32)>, range: Range<i32>) -> (i32, i32) {
    let mut rng = rand::thread_rng();
    'outer: loop {
        let random_x = rng.gen_range(range.clone());
        let random_y = rng.gen_range(range.clone());
        for (x, y) in points.iter() {
            if (x - random_x).abs() < 40 && (y - random_y).abs() < 40 {
                continue 'outer;
            }
        }
        return (random_x, random_y);
    }
}

#[derive(Debug, Copy, Clone)]
struct NodeData {
    x: i32,
    y: i32,
}
#[derive(Debug, Copy, Clone, Default)]
struct EdgeData {
    dist: f64,
    pheromone: f64,
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

fn initialize_graph(points: &VecDeque<(i32, i32)>) -> UnGraph<NodeData, EdgeData> {
    let mut graph = Graph::new_undirected();
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
                    EdgeData {
                        dist,
                        ..Default::default()
                    },
                );
            }
        }
    }
    assert!(is_fully_connected(&graph)); // making sure that the graph is fully connected
    graph
}

const FPS: u32 = 30;
const MAX_POINTS: usize = 100;
const WINDOW_TITLE: &str = "Traveling Salesman Problem";

#[derive(Debug, Clone, Default)]
struct Ant {
    path: Vec<NodeIndex>,
}

#[derive(Debug)]
struct ACO {
    graph: UnGraph<NodeData, EdgeData>,
    start_node: NodeIndex,
    alpha: f64,
    beta: f64,
    ants: Vec<Ant>,
}

#[derive(Debug)]
enum State {
    Populating,
    InitGraph,
    MakingPath(ACO),
}

fn main() {
    let mut window: PistonWindow = WindowSettings::new(WINDOW_TITLE, [1100, 1100])
        .samples(4)
        .build()
        .unwrap();
    let mut points = VecDeque::new();
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
                    state = State::InitGraph;
                }
            }
            State::InitGraph => {
                let graph = initialize_graph(&points);
                // randomly pick a node to start with
                let mut rng = rand::thread_rng();
                let start_node = graph
                    .node_indices()
                    .nth(rng.gen_range(0..graph.node_count()))
                    .unwrap();
                let aco = ACO {
                    graph,
                    start_node,
                    alpha: 1.0,
                    beta: 1.0,
                    ants: vec![Ant::default(); 10],
                };
                state = State::MakingPath(aco);
            }
            State::MakingPath(aco) => {}
        }

        cc.draw_series(points.iter().map(|(x, y)| Circle::new((*x, *y), 10, RED)))
            .unwrap();

        std::thread::sleep(std::time::Duration::from_millis(1000 / FPS as u64));
        Ok(())
    };
    while draw_piston_window(&mut window, &mut f).is_some() {}
}
