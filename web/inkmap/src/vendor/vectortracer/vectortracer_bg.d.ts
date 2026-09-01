/* tslint:disable */
/* eslint-disable */
/**
*/
export function main(): void;
export function __wbg_set_wasm(exports: unknown): void;
export interface RawImageData {
    data: Uint8ClampedArray;
    width: number;
    height: number;
}

export interface BinaryImageConverterParams {
    debug: boolean | undefined;
    mode?: 'polygon'|'spline'|'none';
    cornerThreshold?: number;
    lengthThreshold?: number;
    maxIterations?: number;
    spliceThreshold?: number;
    filterSpeckle?: number;
    pathPrecision?: number;
}

export interface Options {
    invert: boolean | undefined;
    pathFill: string | undefined;
    backgroundColor: string | undefined;
    attributes: string | undefined;
    scale?: number;
}

/**
*/
export class BinaryImageConverter {
  free(): void;
/**
* @param {ImageData} imageData
* @param {BinaryImageConverterParams} converterOptions
* @param {Options} options
*/
  constructor(imageData: ImageData, converterOptions: BinaryImageConverterParams, options: Options);
/**
*/
  init(): void;
/**
* @returns {boolean}
*/
  tick(): boolean;
/**
* @returns {string}
*/
  getResult(): string;
/**
* @returns {number}
*/
  progress(): number;
}
